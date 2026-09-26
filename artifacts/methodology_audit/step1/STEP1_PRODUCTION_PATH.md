# Step 1 — Trace the actual production path

Protocol: `docs/METHODOLOGY_AND_REPORTS.md`, item 1.
Run: 2026-09-15, commit `3e9dbcd` (working tree clean apart from untracked
`data/raw/external_ratings_*.json` and the protocol file), Python 3.11.15,
numpy 1.26.3, scipy 1.16.3, macOS (Darwin 25.3.0).

Scope decision (agreed before execution): the repository has three
methodological layers, traced as three parallel sub-traces and NOT collapsed
into one PASS/FAIL. Three questions are kept distinct throughout:

1. **Freshness** — does a clean rebuild from current code reproduce the committed artifact?
2. **Current-code reproducibility** — does the documented command run from a clean checkout and give a stable result?
3. **Methodology trace** — does the code path match what the documentation claims? (Deferred to Steps 2–4, which audit each stage's mathematics.)

## Gate table

| Path | User-facing? | Research claim? | Current artifact | Rebuild matches? | Documented command runs? | Methodology trace | Gate |
|---|---|---|---|---|---|---|---|
| 1 MODEL (live fitted-model tab) | Yes | Yes | `docs/data/training.json` | **PASS** — byte-identical | PASS | pending Steps 2, 13 | PASS (freshness + reproducibility) |
| 2 Optimized (the shipped bracket) | Yes | Yes | `artifacts/candidates/candidates_2026.json` → `docs/data/season_*.json`, `seasons.json` | **PASS** — identical except `meta.generated_at`; all 17 season payloads byte-identical | PASS | pending Steps 2–6 | PASS (freshness + reproducibility) |
| 3 Pool-aware research (`meta_region_poolaware`) | No | Yes | `artifacts/headline_measurement/canonical_2011_2025_n14.txt` | **PASS** — all 14 evaluation-season rows and the aggregate block identical | **FAIL as documented → fixed → PASS** (see F-1) | pending Steps 4–9, 15–16 | PASS after doc fix; F-1 recorded |

## Evidence

### Path 1 — MODEL

- UI entry: `docs/app.js` (`MODEL` strategy) → `docs/fit.js` (`fitLinear`, `calibrate`).
- Only artifact the browser fetches for this path: `docs/data/training.json`
  (`app.js:143`). `training_pit.json` / `eval_pit_tournament.json` are research
  inputs for `scripts/experiment_model_families.py`, not loaded by the page.
- Rebuild: `python3 scripts/build_training_matrix.py` (writes in place).
  Committed copy saved to `/tmp` first, then `cmp` → **IDENTICAL**; `git status`
  clean afterwards. 1,008 games, 16 seasons (2010–2026), 31 variables.

### Path 2 — Optimized

- UI entry: `docs/app.js:285` reads `state.season.pool_optimized`; the page
  fetches only `data/season_${year}.json`, `data/training.json`, `data/seasons.json`.
- Chain: `scripts/experiments/build_candidate_artifact.py --year 2026` →
  `artifacts/candidates/candidates_2026.json` → `src/product/selection.py`
  (product.v3 selector, frozen objectives `ev`/`p1`) via
  `scripts/build_ui_payload.py` → `docs/data/season_2026.json` (`pool_optimized`).
- Candidate rebuild (73 s, `--out /tmp/...`, no tracked file touched):
  sha256 differs (`18ad9519…` committed vs `466ea181…` rebuilt); deep semantic
  compare of every top-level key equal after dropping `meta.generated_at`.
  Classification: **expected / non-semantic** (timestamp only).
- Payload rebuild: `python3 scripts/build_ui_payload.py` regenerates all 17
  `season_*.json` and `seasons.json` in place. All **byte-identical** to the
  backed-up committed copies; `git status docs/data/` empty.
- Side finding S-1 (not a gate item): `docs/data/candidates_2026.json` is a
  stale second copy (last commit `261eaa1`; the live `artifacts/candidates/`
  copy has 6 later commits; 473,882 canonicalised diff lines). `app.js` never
  fetches it. Cleanup candidate, not a live-path defect. [Pre-commit review
  2026-09-15: it is a deliberate committed CI fixture (the real artifacts are
  gitignored) pinned by content in `test_material_difference.py`; kept as is,
  annotated in `test_selection_sunday_rehearsal.py`.]

### Path 3 — Pool-aware research

- Claim under trace: README "What the backtest number means" —
  `meta_region_poolaware` P(1st) 12.0% vs `seed` 4.0%, n=14 seasons.
- **F-1 (protocol L, reproducibility): the documented command did not run.**
  README (lines 54, 63, 168, 251), `.claude/agents/pool-optimizer/CLAUDE.md`,
  and `.claude/skills/pool-optimizer-backtest.md` all gave
  `python scripts/mc_pool_backtest.py …`. From the repo root with a stock
  interpreter this fails at `scripts/mc_pool_backtest.py:31`:

  ```
  ModuleNotFoundError: No module named 'scripts'
  ```

  Cause: `from scripts._common import …` executes before the file's own
  `sys.path.insert` (line 39). `python file.py` puts `scripts/` on `sys.path`,
  not the repo root, so the `scripts` package is unresolvable. No editable
  install `.pth` for this project exists in site-packages; `PYTHONPATH` unset.
  `python -m scripts.mc_pool_backtest` with identical arguments runs.
  Classification: **unambiguous packaging/documentation defect; not evidence
  about the methodology.** Fix applied (docs only, after the failure was
  recorded): all six documented occurrences changed to the `-m` form. Smoke
  test of the corrected form under `env -i` (no `PYTHONPATH`) runs.
- Rebuild (corrected command, exact README arguments, `--no-log`), output saved
  as `path3_rebuild_mc_pool_backtest.txt` beside this file. Compared with the
  committed artifact:
  - per-season result rows, 2011–2025 (28 rows): **identical**
  - `selected=` candidate and P1 for every season: **identical**
  - AGGREGATE block (seed 0.0399 ± 0.0079; meta 0.1200 ± 0.0336; MeanRank
    paired t=13.167, 14/14; BestRank t=2.859, 11/14): **identical**
  - The run is deterministic under CRN; the match is exact, not "within noise".
- Two non-result differences, both classified:
  - D-1 `Years: 15` (rebuild) vs `Years: 14` (artifact header). `BACKTEST_YEARS`
    has included 2026 since commit `6fb877a` (2026-04-16); the aggregate strips
    it as an integration season (README documents this). The extra 2026 rows
    do not enter any reported figure. Classification: **expected / non-semantic.**
  - D-2 2018 `best of 26` → `best of 23` (selected candidate and P1 unchanged;
    all other seasons' counts identical). Diagnosed by wrapping
    `construct_bracket` at runtime (no code edit): 44 calls, 0 exceptions, 23
    unique after exact-bracket dedup. So three constructions now coincide
    with others; nothing failed. Classification: **changed code since
    2026-09-10, result-neutral.** Attributable to one of `bc4519d`, `4908b91`,
    `8cc9b85`, `bcd4ded`; not bisected because the outcome is unaffected.
- Follow-ups carried forward (not acted on, per protocol C/I):
  - `scripts/mc_pool_backtest.py:4070` `except Exception: pass` around
    candidate construction is a silent fallback. Zero failures today, but it
    would hide a broken base without any log line. → Step 5 (candidate-space
    integrity).
  - Seven sibling scripts share the same import-before-path-fix pattern and
    would fail as `python scripts/<x>.py`: `ablation_seed_features`,
    `backtest_ev_vs_kaggle`, `backtest_pool_strategy`, `divergence_diagnostic`,
    `feature_seed_correlation`, `validate_e8_interactions`,
    `unified_mode_evaluation`. None is documented in the broken form, so no
    doc change was needed; a code-level fix is out of scope for this step.

## Downstream impact

- F-1 affects reproducibility of Path 3 only. It does not touch Paths 1 or 2,
  whose commands ran as documented, and it says nothing about whether
  `meta_region_poolaware` is statistically sound (Steps 4–9).
- S-1 has no downstream impact on any gate.
- D-1, D-2: none.

## Commands to reproduce this step

```bash
python3 scripts/build_training_matrix.py && git status --short docs/data/training.json
python3 scripts/experiments/build_candidate_artifact.py --year 2026 --out /tmp/rc
python3 scripts/build_ui_payload.py && git status --short docs/data/
python -m scripts.mc_pool_backtest --team-identity --opponent pool \
  --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware --no-log
```
