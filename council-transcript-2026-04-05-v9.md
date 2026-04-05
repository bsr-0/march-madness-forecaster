# LLM Council Transcript — Session #14
**Date:** April 5, 2026
**Topic:** Holistic Repository Audit — Critical Gap Identification

---

## Original Question

> Analyze the repository holistically — codebase, architecture, test coverage, security, documentation, and developer experience. Cross-reference any findings or notes from recent council transcripts. Identify the single most critical gap that, if left unaddressed, poses the highest risk or largest drag on project quality.

---

## Framed Question

**What is the single most critical unresolved gap in the march-madness-forecaster repository?**

Context: The project has pivoted from ML prediction to pool game theory optimization. Recent audits purged 47 contaminated data files and added LeakageError infrastructure throughout the pipeline. However, `src/prediction/noseed_model.py:_load_team_stats()` still loads Torvik files without calling `_validate_pretournament()` — a guard that exists in `scripts/mc_pool_backtest.py` but not in the production module. This bypasses the entire LeakageError exception hierarchy. No test currently covers this path.

---

## Advisor Responses

### The Contrarian

The missing `_validate_pretournament()` call is real, but stop congratulating yourselves for finding it — it's a symptom, not the disease. The actual critical gap is this: there is no automated test that would catch this class of bug. You have 724 tests collected but the dependency chain is broken. Adding the call fixes today's known hole. It does nothing for the next `_load_something()` method someone adds in three months that also skips validation. The real fix is a leakage integration test: load a fixture with `data_type = "post_tournament"`, run it through `noseed_model._load_team_stats()`, and assert that it raises. That test would have caught this bug before the first backtest ever ran. Without that test, you're playing whack-a-mole with guard clauses.

### The First Principles Thinker

The missing `_validate_pretournament()` call isn't a bug. It's a broken contract at the system's existential boundary. The leakage prevention machinery exists precisely because this project's authors understood that silent contamination is catastrophic. Then they built a production module that bypasses it. The fix isn't adding a guard. The fix is making it architecturally impossible to load team stats without the temporal check. The validation should live in the data loading layer, not be caller-dependent. Until that's resolved, every backtest result from the noseed model carries an unquantified contamination risk.

### The Expansionist

The validation gap is real, but fix it properly to unlock verified, auditable backtest provenance. Add `_validate_pretournament()` to `noseed_model._load_team_stats()`, yes. But wrap it in a lightweight data lineage layer: each model run logs what files were loaded, their modification timestamps, and whether they passed the pre-tournament gate. Credibility as a product — a bracket optimizer that produces signed, reproducible results is a fundamentally different offering.

### The Outsider

This entire setup assumes the person running it knows which file is "pre-tournament." How does that `data_type` field get set in the first place? If it's set manually, or inferred from filename conventions, or written during a data pipeline step that has no tests — then the guard, even when present, is checking a value that could be wrong. You'd have a validated lie. The deeper gap is: there is no verified, automated data provenance chain from raw Torvik download to the `data_type` field that the guard reads.

### The Executor

The fix is a 15-minute task. Open `noseed_model.py`. Find `_load_team_stats()`. Add one call to `_validate_pretournament()`. Done. Write one test that passes a post-tournament file and confirms it raises. Run pytest and ruff. Ship it.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Expansionist
- Response B = The Contrarian
- Response C = The Outsider
- Response D = The Executor
- Response E = The First Principles Thinker

### Consensus: Response B (Contrarian) rated strongest by all 5 reviewers.

### Blind Spots Caught by Peer Review

1. **Call graph untraced**: No reviewer asked who calls `_load_team_stats()` in production CLI/API. There are two call sites in `src/cli/pool_cmds.py` (lines 168 and 193) — the guard was missing on the live production path, not just backtesting.

2. **data_type provenance unverified**: The guard checks a field that may be set by convention or manual annotation. Reviews 1 and 4 flagged this independently.

3. **Guard correctness unaudited**: If `_validate_pretournament()` has false-negative paths, the architectural fix locks in a flawed invariant.

---

## Chairman Verdict

### Where the Council Agrees

The `_validate_pretournament()` call is missing from `noseed_model._load_team_stats()` and must be added. The fix without a regression test is not a fix — it's a one-time patch that leaves the next bypass invisible. Both must ship as a single atomic unit.

### Where the Council Clashes

B (test coverage gap), E (architectural design failure), and C (data_type provenance) are all correct at different abstraction layers. B is right about what to do this week. E is right about the long-term architecture. C is right that neither fully closes the exposure.

A's data lineage audit trail vision is correct in direction but wrong in sequence — don't build provenance infrastructure on a foundation that isn't yet verified.

### The Recommendation

Add `_validate_pretournament()` to `_load_team_stats()` and write the integration test as a single atomic commit. The test must cover both rejection (post-tournament fixture → LeakageError) and acceptance (pre-tournament fixture → returns data). Read the guard's own logic before building further on it.

### The One Thing to Do First

Grep the full call graph of `_load_team_stats()` to know the blast radius — then fix and test as one unit.

---

## Resolution

**Implemented in this session (2026-04-05):**

1. Added `_validate_pretournament()` function to `src/prediction/noseed_model.py` with `LeakageError` on missing or wrong `data_type`
2. Called it in `_load_team_stats()` after loading both `torvik_{year}.json` and `torvik_four_factors_{year}.json`
3. Added `tests/test_noseed_model_leakage.py` with 10 regression tests covering rejection, acceptance, and both file types
4. All 10 new tests pass; `ruff check src/` clean
