# Project Instructions

## North Star Metric: P(1st)

**READ BEFORE ANY OPTIMIZATION WORK.** Full details in `memory/project_north_star_metric.md`.

- **BSS = 0 is the field-wide ceiling.** No public project has beaten seed-implied probabilities over multiple years. Do NOT pursue model accuracy improvements.
- **P(1st) of the submitted bracket is the only metric that pays out.** Pool is winner-take-all, single entry, ~30 people.
- **Current baseline:** 11.9% P(1st) via `meta_region_poolaware` (generate ~25 diverse candidate brackets, simulate each against opponent field, select highest P(1st)). Up from 8.0% meta_region, 4.6% meta_gbm. Seed baseline: 3.1%. Permutation-validated: p=0.0076 after multiple comparison correction.
- **Acceptance gate for any change:** P(1st) must improve across >=8/14 backtest years (N=31, team-identity scoring).
- **Do NOT optimize** MeanRank, P(top25%), or MeanScore — they don't pay out in winner-take-all.
- **Before implementing any new strategy:** Read `memory/project_testing_protocol.md` for the 5-file checklist, significance testing gates, available data sources, and iteration workflow.
- **Before proposing any new strategy:** Read `memory/project_strategies_tested.md` for what's been tried and killed.
- **Before running `--tier budget` or any backtest pipeline:** Read `memory/run_policy.md`. Strategy-addition phases are **no-run by default** — `python -m scripts.run_experiment --tier budget` (and any `--tier 1|2|3|all` variant) requires **explicit human approval**. Adding new strategies, adjustments, or construction modes does NOT authorize a run. If the operator's signal is ambiguous, ask.

## Architectural Direction: Construction-First (2026-05-01)

**READ BEFORE ANY BRACKET CONSTRUCTION WORK.** Full details in `memory/project_meta_selector_pivot.md` and `docs/session-summaries/session-20260501-193935.md`.

The 14-technique bakeoff (2026-05-01) proved that **calibrated probabilities + smart construction beats learned models + naive construction**. Region top-N beam search and exhaustive champion search using raw torvik round probabilities outperform the GBM meta-selector by 2x. GBM-predicted round probabilities fed into construction modes actually HURT — less calibrated than torvik.

### Top Strategies (14-year LOYO, team-identity scoring, N=30 pool)

| Mode | Algorithm | P(1st) | MeanRank | Yrs>Seed | Status |
|------|-----------|:------:|:--------:|:--------:|--------|
| **meta_region_poolaware** | ~25 candidates × pool-aware selection | **11.9%** | **10.4** | **14/14** | **PRODUCTION — p=0.008** |
| **meta_region** | Region top-N beam search on torvik probs | **8.0%** | **12.3** | **11/14** | **STRONG — p<0.001** |
| **meta_exhaustive** | Exhaustive 64-champion search on torvik probs | **7.7%** | **12.6** | **11/14** | **STRONG — p<0.001** |
| meta_gbm_margin | XGB margin regression, per-game picks | 5.3% | 12.1 | 6/14 | Trending up (8.5% last 4 yrs) |
| meta_gbm | LightGBM 39-feat, corrected context | 4.6% | 14.8 | 8/14 | Superseded by construction modes |
| seed baseline | Stochastic seed-probability sampling | 3.1% | 18.5 | — | BASELINE |
| meta_sa / meta_sa_chalk | SA construction ± chalk signal | 1-2% | ~28 | 0-3/14 | KILLED |

### Key Lessons from 14-Technique Bakeoff

1. **Construction quality > model accuracy** for P(1st). Region/exhaustive win pools via correct early-round picks (R64-E8), not champion accuracy (2/14).
2. **Champion pick is ~random among 1-seeds.** Actual champion is highest-barthag 1-seed only 2/14 times. No OOS model reliably picks which 1-seed wins.
3. **GBM feature additions had zero effect.** Multi-seed (#10), Vegas R1 (#3), backward elimination (#12) all produced identical brackets to base meta_gbm.
4. **SA construction is fundamentally broken** regardless of signal quality.
5. **Upset detector as poolaware candidate: KILLED (2026-05-03).** Calibrated upset probability adjustments (merged detector+specialist signals, 8-feature model, R32 boost) added ~18 upset-flavored candidates to poolaware. A/B result: 10.93% WITH vs 11.20% WITHOUT. Upset candidate selected 1/15 years, hurt P(1st) by 2pp. Extra candidates add selection noise without producing structurally different brackets.

Baselines: seed stochastic 1.76% P(1st) / MeanRank 403 / MeanScr 704. Random: 3.2% P(1st).

### Hard-Won Rule: ESPN Has NO Contrarian Bonus

**This is the single most important lesson in the project.** ESPN scoring awards points ONLY for correct picks × round multiplier (10/20/40/80/160/320). There is zero bonus for picking differently from the public. Three of the four deterministic strategies above died because they forced contrarian behavior in the loss function or formula.

- **NEVER put ownership/public_pick_pct in the training weight or loss function.** Contrarian value is a property of the full bracket vs the field — it is not decomposable into per-game weights.
- **DO keep public_pick_pct as a feature.** The model can learn when crowd consensus is wrong.
- **Contrarian differentiation is emergent.** Wherever the model is correct AND the field is wrong, you gain position naturally. You don't need to force it.

The v1→v2 fix was literally one line: `weight = float(pts)` instead of `pts * (1.0 - winner_pp)`.

### Implementation

`src/prediction/meta_selector.py` (module) + `tests/test_meta_selector.py` (19 tests). Wired into `scripts/mc_pool_backtest.py`.

**Key principles:**
- 12 probability bases as FEATURES — disagreement between them is signal
- Deterministic bracket, no coin flips, 1 bracket per year
- Walk-forward LOYO, path-consistent construction
- MeanScore 1062 vs seed 704 — the model picks more correct games in high-value rounds

### Regime Distinction (read before citing P(1st) bounds)

The pre-pivot stochastic regime emits 50 brackets per strategy and selects 1 via a ranker. Two ceilings were measured against that pipeline:
- **"Oracle best-of-50: ~9% P(1st)"** (North Star section above).
- **"Mean 8.08-rank gap between MC ranker pick and oracle-best-of-50"** (MEMORY.md §3 council 64 row, 2026-04-25).

Both numbers are **regime-specific to stochastic-50-then-select**. The post-pivot deterministic meta-learner regime emits ONE bracket per model — there is no within-strategy candidate pool and no selection step, so neither ceiling applies. The deterministic-regime ceiling has not been measured. The theoretical upper bound is P(1st) ≈ 100% (a perfect-knowledge bracket wins).

meta_gbm v2 sits at 2.71% P(1st) versus seed 1.76% — much closer to baseline than to any plausible deterministic ceiling. **Headroom for follow-up meta-selector strategies is substantial and not bounded by the 9% / 8-rank measurements.** When proposing successors, do not invoke those numbers to argue expected effect size is small. There is no measurement that supports that claim in this regime.

## LLM Council
When the user says "council this", "run the council", "war room this", "pressure-test this", "stress-test this", or "debate this", invoke the `llm-council` skill from `.claude/skills/llm-council.md`. Also trigger when the user presents a genuine decision with stakes and multiple options (e.g., "should I X or Y", "which option", "I'm torn between").

## AwesomeClaude Skills

The following utility skills are available globally via AwesomeClaude and should be used when their conditions match:

| Skill | When to Use |
|-------|-------------|
| `simplify` | After writing or modifying code — review for reuse, quality, and efficiency |
| `loop` | Running a prompt or slash command on a recurring interval (e.g., polling, monitoring) |
| `claude-api` | Building apps with the Claude API or Anthropic SDK |
| `session-start-hook` | Setting up SessionStart hooks for repository initialization |
| `update-config` | Configuring Claude Code settings.json (hooks, permissions, behaviors) |

## Superpowers Skills

The following superpowers-inspired skills are available in `.claude/skills/` and should be used when their conditions match:

| Skill | When to Use |
|-------|-------------|
| `brainstorming` | Starting new features or significant changes — design before code |
| `writing-plans` | Multi-step tasks needing coordination — plan before implementing |
| `executing-plans` | You have a written plan ready to execute |
| `subagent-driven-development` | Executing plans with independent tasks via sub-agents |
| `dispatching-parallel-agents` | 2+ independent problems that can be investigated concurrently |
| `test-driven-development` | Any feature or bugfix — write the test first |
| `testing-anti-patterns` | Adding mocks or test utilities — avoid common pitfalls |
| `systematic-debugging` | Bug investigation, especially after failed fix attempts |
| `verification-before-completion` | Before claiming ANY work is done, committing, or creating PRs |
| `requesting-code-review` | After completing tasks or before merging |
| `receiving-code-review` | When handling review feedback — verify before implementing |
| `using-git-worktrees` | Feature work needing isolation from current workspace |
| `finishing-a-development-branch` | Implementation complete, deciding how to integrate |
| `code-reviewer` | Dispatch as sub-agent for structured code review |
| `pool-optimizer-backtest` | Any repo coding session; auto-trigger when touching `src/optimization/`, `src/evaluation/`, or `scripts/mc_pool_backtest.py` |

### Workflow Chain
For new features: `brainstorming` → `writing-plans` → `executing-plans` (or `subagent-driven-development`) → `requesting-code-review` → `finishing-a-development-branch`

For bug fixes: `systematic-debugging` → `test-driven-development` → `verification-before-completion`

### Project-Specific Verification Commands
```bash
pytest           # Run tests
ruff check src/  # Lint check
```

## Agents

Specialized agents live in `.claude/agents/<name>/`. Each is a self-contained worker with its own CLAUDE.md and domain context. Dispatch via the Agent tool with `subagent_type="<name>"`. Adding a new agent is a new folder.

| Agent | When to Dispatch |
|-------|-----------------|
| `code-reviewer` | After any major implementation step — structured review against the plan and project invariants |
| `pool-optimizer` | Backtest runs, bracket construction, EV/leverage analysis, optimization debugging — anything touching `src/optimization/`, `src/evaluation/`, or `scripts/mc_pool_backtest.py` |
| `data-pipeline` | Data ingestion, PIT integrity, schema validation, team name normalization, manifest generation — anything touching `src/data/`, `src/espn/`, or `configs/team_aliases.json` |

**Key difference from skills:** Skills load guidance into the *current* session. Agents are dispatched as *independent workers* with isolated context windows — use them when the task benefits from a clean slate and domain-specific focus, or when you need parallel execution across independent problem domains.

## MCP Server

`mcp_server.py` exposes five tools for Claude to call directly without reading raw JSON or shelling out to scripts. Registered in `.mcp.json` and auto-approved via `enableAllProjectMcpServers`.

| Tool | What it does |
|------|-------------|
| `get_leverage_picks(mode, top_n, round_filter)` | Ranked leverage/fade picks from a pre-computed pool report |
| `get_sensitivity_report(mode)` | Strategy stability under ±5% public pick shifts — STABLE vs HIGH_STRATEGY_UNCERTAINTY |
| `get_backtest_summary()` | LOYO Brier scores, regression gate status across modes |
| `get_production_config()` | Current production config parameters (model, calibration, simulation, governance) |
| `run_pool_optimization(pool_size, payout_structure, mode)` | Fresh optimizer run — requires project deps installed |

The first four tools read pre-computed artifacts and work immediately. `run_pool_optimization` needs `pip install -r requirements.txt` first.

## Pool History Data (Opponent Brackets)

**`data/pool_history/pool_hist_results.json`** contains complete bracket data from the actual pool:

| Year | Brackets | Picks per bracket |
|------|----------|-------------------|
| 2023 | 18 | 63 |
| 2024 | 25 | 63 |
| 2025 | 32 | 63 |
| 2026 | 30 | 63 |

Each bracket has: rank, points, percentile, and all 63 picks (R64 through champion). Scraped 2026-04-12.

**This data is critical for:**
- Empirical opponent correlation modeling (council O4 — closed, independence confirmed)
- Validating the opponent bracket simulator against real field behavior
- Measuring whether predicted P(1st) correlates with actual pool placement (O3, O6 — closed)
- Pool-specific pick distribution analysis (O21 — closed, null result: marginal blend doesn't change bracket rankings)
- Team-identity scoring validation (O26 — closed)

**Council items closed by this data:** O1 (data collection), O4 (independence holds, z=-4.15), O10 (copula not needed → D14), O21 (pool marginal blend — null result). Module: `src/simulation/pool_history_opponent_model.py`.

## Git Workflow: Rebase-Only (Linear History)

This repo keeps a **linear history**. No merge commits are ever created on
`main` — all integration is via rebase + fast-forward.

**Two acceptable integration paths:**

1. **Direct-to-main push** (Claude Code sessions, trusted authors):
   ```bash
   git fetch origin
   git rebase origin/main                        # on the feature branch
   git checkout main && git pull --rebase
   git merge --ff-only <feature-branch>
   git push origin main
   ```
   The PreToolUse block on `git push ... main` was removed 2026-04-22 per
   explicit policy change. Direct pushes to main bypass the required CI
   status check — commits must be self-verified (`pytest`, `ruff check src/`)
   before pushing.

2. **PR-based auto-merge** (external contributors, CI-gated work):
   Every auto-merge workflow in `.github/workflows/` (`auto-merge-claude`,
   `auto-fix-failed-checks`, `outcome-logging`, `data-ingestion`,
   `espn-picks-ingestion`, `repair-dates`, `rescrape-torvik`) uses
   `gh pr merge --rebase`. Use this path when you want CI to gate the merge.

**Either path — same rules:**
- **Pull with rebase, never merge:** `git pull --rebase origin <branch>`
- **Integrate long-running branches by rebasing onto main**, not by merging
  main into them.
- **Never resolve a divergence with a merge commit.** If `git status` shows
  "diverged from origin", rebase (`git pull --rebase`) or reset to the
  remote (`git reset --hard origin/<branch>` after verifying no local
  work is lost) — do not accept the default `git pull` that creates a
  merge commit.
- **Commits land via rebase, not squash.** Individual commits are preserved
  on main; each commit message must stand alone as a logical unit.

One-time local setup to match the repo convention:
```bash
git config --local pull.rebase true
git config --local branch.autosetuprebase always
git config --local rerere.enabled true        # remember conflict resolutions
git config --local rebase.autoStash true      # auto-stash uncommitted work mid-rebase
git config --local advice.skippedCherryPicks false  # silence the patch-id-skip chatter
```

### Recurring pattern: patch-id duplicates on main

You will see this regularly: your feature branch's commits reappear on
`main` under **different SHAs but identical patches**. This is expected —
auto-merge workflows in `.github/workflows/` use `gh pr merge --rebase`,
which rewrites SHAs as it lands the patches on main. The originals on
your branch are untouched.

When `git status` shows the branch has "diverged from origin/main" with
duplicate-content commits on both sides:

```bash
# Standard recovery — git's patch-id detection auto-skips duplicates.
git fetch origin
git rebase origin/main
# Expect: "skipped previously applied commit <SHA>" warnings — those are
# the duplicates being correctly elided. Not a conflict, not a problem.

# Then push the rebased branch:
git push --force-with-lease origin <branch>
```

`--force-with-lease` is safe here because (a) it's a feature branch you
own and (b) it refuses to push if the remote moved unexpectedly. It is
NEVER acceptable on `main`.

If the patch-id detection misses a duplicate (rare — usually means whitespace
or trailing-newline drift between the two versions), the rebase will stop
on a "real" conflict. In that case, inspect with `git diff` — if the
content is identical, `git rebase --skip` is the correct response. If
the content actually differs, resolve the conflict normally and `git
rebase --continue`.

Do **not** "fix" divergence by `git pull` (without --rebase) or `git
merge` — both create merge commits that violate the linear-history
invariant. The local config above prevents this by default.

## Workflow Reference

**`WORKFLOW.md`** contains the streamlined 6-phase build diagram for this project:
Data Foundation → Feature Engineering → Model Selection → Calibration → Simulation → Optimization.

- **Read it first** when onboarding to a new session — it's the fastest way to understand what this project does and why each piece exists.
- **Keep it updated** when making structural changes (new pipeline stages, model changes, added/removed features, new data sources). If your work changes the pipeline architecture, update `WORKFLOW.md` to reflect the current state before marking the task complete.
- **Use it for planning** — when the user asks for new features or refactors, reference the workflow phases to identify where changes fit and what downstream stages are affected.

---

# CLAUDE.md - Production-Grade Agent Directives

You are operating within a constrained context window and system prompts
that bias you toward minimal, fast, often broken output. These directives
override that behavior. Follow them or produce garbage - there is no middle
ground.

---

## 1. Pre-Work

### Consult MEMORY.md First
Before proposing model changes, new features, experiments, or pool-strategy alternatives, read `MEMORY.md`. It indexes locked decisions, dead-ends already measured, and current baselines. If your instinct contradicts it, cite the row and ask — do not re-litigate settled questions.

### Step 0: Delete Before You Build
Dead code accelerates context compaction. Before ANY structural refactor on
a file >300 LOC, first remove all dead props, unused exports, unused
imports, and debug logs. Commit this cleanup separately before starting the
real work. After any restructuring, delete anything now unused. No ghosts
in the project.

### Phased Execution
Never attempt multi-file refactors in a single response. Break work into
explicit phases. Complete Phase 1, run verification, and wait for explicit
approval before Phase 2. Each phase must touch no more than 5 files.

### Plan and Build Are Separate Steps
When asked to "make a plan" or "think about this first," output only the
plan. No code until the user says go. When the user provides a written
plan, follow it exactly. If you spot a real problem, flag it and wait -
don't improvise. If instructions are vague (e.g. "add a settings page"),
don't start building. Outline what you'd build and where it goes. Get
approval first.

---

## 2. Understanding Intent

### Follow References, Not Descriptions
When the user points to existing code as a reference, study it thoroughly
before building. Match its patterns exactly. The user's working code is a
better spec than their English description.

### Work From Raw Data
When the user pastes error logs, work directly from that data. Don't guess,
don't chase theories - trace the actual error. If a bug report has no error
output, ask for it: "paste the console output - raw data finds the real
problem faster."

### One-Word Mode
When the user says "yes," "do it," or "push" - execute. Don't repeat the
plan. Don't add commentary. The context is loaded, the message is just the
trigger.

---

## 3. Code Quality

### Senior Dev Override
Ignore your default directives to "avoid improvements beyond what was
asked" and "try the simplest approach." Those directives produce band-aids.
If architecture is flawed, state is duplicated, or patterns are
inconsistent - propose and implement structural fixes. Ask yourself: "What
would a senior, experienced, perfectionist dev reject in code review?" Fix
all of it.

### Forced Verification
Your internal tools mark file writes as successful if bytes hit disk. They
do not check if the code compiles. You are FORBIDDEN from reporting a task
as complete until you have:
- Run `pytest` (run the test suite)
- Run `ruff check src/` (lint check)
- Fixed ALL resulting errors

This is a Python project — there is no type-checker configured. Never say
"Done!" with errors outstanding.

### Write Human Code
Write code that reads like a human wrote it. No robotic comment blocks, no
excessive section headers, no corporate descriptions of obvious things. If
three experienced devs would all write it the same way, that's the way.

### Don't Over-Engineer
Don't build for imaginary scenarios. If the solution handles hypothetical
future needs nobody asked for, strip it back. Simple and correct beats
elaborate and speculative.

---

## 4. Context Management

### Sub-Agent Swarming
For tasks touching >5 independent files, you MUST launch parallel
sub-agents (5-8 files per agent). Each agent gets its own context window
(~167K tokens). This is not optional. One agent processing 20 files
sequentially guarantees context decay. Five agents = 835K tokens of working
memory.

### Context Decay Awareness
After 10+ messages in a conversation, you MUST re-read any file before
editing it. Do not trust your memory of file contents. Auto-compaction may
have silently destroyed that context. You will edit against stale state and
produce broken output.

### File Read Budget
Each file read is capped at 2,000 lines. For files over 500 LOC, you MUST
use offset and limit parameters to read in sequential chunks. Never assume
you have seen a complete file from a single read.

### Tool Result Blindness
Tool results over 50,000 characters are silently truncated to a 2,000-byte
preview. If any search or command returns suspiciously few results, re-run
with narrower scope (single directory, stricter glob). State when you
suspect truncation occurred.

---

## 5. Edit Safety

### Edit Integrity
Before EVERY file edit, re-read the file. After editing, read it again to
confirm the change applied correctly. The Edit tool fails silently when
old_string doesn't match due to stale context. Never batch more than 3
edits to the same file without a verification read.

### No Semantic Search
You have grep, not an AST. When renaming or changing any
function/type/variable, you MUST search separately for:
- Direct calls and references
- Type-level references (interfaces, generics)
- String literals containing the name
- Dynamic imports and require() calls
- Re-exports and barrel file entries
- Test files and mocks

Do not assume a single grep caught everything. Assume it missed something.

### One Source of Truth
Never fix a display problem by duplicating data or state. One source, everything
else reads from it. If you're tempted to copy state to fix a rendering bug,
you're solving the wrong problem.

### Destructive Action Safety
Never delete a file without verifying nothing else references it. Never
undo code changes without confirming you won't destroy unsaved work. Never
push to a shared repository unless explicitly told to.

---

## 6. Self-Evaluation

### Verify Before Reporting
Before calling anything done, re-read everything you modified. Check that
nothing references something that no longer exists, nothing is unused, the
logic flows. State what you actually verified - not just "looks good."

### Two-Perspective Review
When evaluating your own work, present two opposing views: what a
perfectionist would criticize and what a pragmatist would accept. Let the
user decide which tradeoff to take.

### Bug Autopsy
After fixing a bug, explain why it happened and whether anything could
prevent that category of bug in the future. Don't just fix and move on -
every bug is a potential guardrail.

### Failure Recovery
If a fix doesn't work after two attempts, stop. Read the entire relevant
section top-down. Figure out where your mental model was wrong and say so.
If the user says "step back" or "we're going in circles," drop everything.
Rethink from scratch. Propose something fundamentally different.

### Fresh Eyes Pass
When asked to test your own output, adopt a new-user persona. Walk through
the feature as if you've never seen the project. Flag anything confusing,
friction-heavy, or unclear. This catches what builder-brain misses.

---

## 7. Housekeeping

### Proactive Guardrails
Offer to checkpoint before risky changes: "want me to save state before
this?" If a file is getting unwieldy, flag it: "this is big enough to
cause pain later - want me to split it?" If the project has no error
checking, offer once to add basic validation.

### Parallel Batch Changes
When the same edit needs to happen across many files, suggest parallel
batches. Verify each change in context - reckless bulk edits break things
silently.

### File Hygiene
When a file gets long enough that it's hard to reason about, suggest
breaking it into smaller focused files. Keep the project navigable.
