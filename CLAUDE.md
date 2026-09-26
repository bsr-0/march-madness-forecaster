# Repository agent instructions

Correctness matters more than speed in this modeling and optimization repository. This file is a short local workflow guide; project decisions and technical standards live in the linked canonical documents.

## Mission and current objective

**Mission:** Build and maintain a reliable NCAA tournament bracket forecaster that helps users choose brackets for specific pool formats, with claims supported by reproducible evidence.

**Current objective:** Deliver and evaluate the 2027 bracket release under the frozen roadmap. Follow the roadmap's active phase and gates; do not treat the current checkpoint as permanent project scope. As of 2026-09-25, the historical promotion evidence is indeterminate, so the frozen seed-only fallback is selected for now. The actual 2027 release and post-tournament evaluation remain pending; do not claim either is complete.

## Which guidance to follow

- The user's current request defines the task and requested actions.
- [PROJECT_ROADMAP_2027.md](docs/PROJECT_ROADMAP_2027.md) controls 2027 release scope, phases, gates, and decisions.
- [METHODOLOGY_AND_REPORTS.md](docs/METHODOLOGY_AND_REPORTS.md) controls methodology and validation for audits and roadmap-required checks.
- [OPERATIONS.md](docs/OPERATIONS.md) documents current operational contracts.
- This file governs the local agent workflow. Archived reports are historical evidence, not current instructions.

If applicable instructions materially conflict, do not silently reconcile them. Explain the conflict and ask for a decision when it affects scope, methodology, data, or release behavior.

## Default workflow

1. Identify whether the request is to explain, plan, review, implement, or perform an operational action; honor that mode.
2. For changes, inspect relevant files and Git status first. Preserve pre-existing work.
3. Bound the task by its requested outcome. Clarify consequential ambiguity; don't add ceremony to small, clear tasks.
4. For behavior changes, trace relevant inputs, callers, outputs, contracts, and targeted tests before editing. Prefer the smallest complete fix.
5. Run the smallest relevant verification. For documentation-only changes, check links and consistency; avoid unrelated test suites.
6. Report changes, actual verification results, and remaining limitations accurately.

## Bug triage and scope

- Verify a suspected defect and trace its affected path before treating it as a bug.
- Fix confirmed defects that block the requested outcome or an active roadmap gate; fix the root cause and verify affected behavior.
- For confirmed defects outside the task/gate, record the evidence and impact and park them. Do not silently expand the task or block unrelated work unless there is a concrete material risk.
- Never mask failures with silent fallbacks or patch only a displayed result.

For 2027 release work, check the roadmap's active phase before starting research. Work one stream at a time; predeclare comparison criteria; stop at the gate or timebox and record PASS, FAIL, or INDETERMINATE. Do not treat the methodology document as an automatic mandate for a full re-audit.

Do not stage, commit, publish, delete, or move user data unless explicitly requested.

## Canonical references

- [README.md](README.md) — project overview and quick start
- [PROJECT_ROADMAP_2027.md](docs/PROJECT_ROADMAP_2027.md) — release plan and decision log
- [METHODOLOGY_AND_REPORTS.md](docs/METHODOLOGY_AND_REPORTS.md) — audit and report standards
- [OPERATIONS.md](docs/OPERATIONS.md) — operational contracts
