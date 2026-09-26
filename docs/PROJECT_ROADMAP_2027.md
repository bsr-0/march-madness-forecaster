# 2027 prospective release roadmap

## Goal

Deliver a reproducible, documented 2027 bracket artifact before the tournament, then evaluate it after the tournament using rules chosen before outcomes are known.

This is the controlling schedule and scope for 2027 release work. It is deliberately bounded: the goal is a defensible release, not an exhaustive re-audit or an open-ended search for the best possible strategy.

## Working rules

- Work on one active workstream at a time.
- Every investigation must state its decision question, release relevance, timebox, and exit criterion before it starts.
- Investigate release-critical risks first. An issue is release-critical only when evidence shows it can invalidate the released artifact or its evaluation.
- Use the methodology protocol for the checks required by the active phase. The existence of an unchecked protocol item does not make it a release blocker.
- Predeclare metrics, baselines, thresholds, and comparison sets before examining results. Do not change them after seeing results without recording the change and its consequences.
- Allow one initial investigation and one follow-up per question. At the timebox limit, record PASS, FAIL, or INDETERMINATE and make the phase decision.
- Put interesting but non-blocking questions in the parking lot. Do not let them silently become prerequisites.
- Reopen a completed decision only for new material evidence, a confirmed defect, or a changed release requirement.
- A failed release gate blocks dependent work; it does not justify restarting unrelated phases.

## Phases and gates

Timeboxes are maximum working time, not minimum duration. They may be shortened when the exit criterion is met. Any extension requires a recorded reason and a revised decision date.

| Phase | Maximum timebox | Work | Exit gate |
|---|---:|---|---|
| 0. Set the contract | 1 week | Specify the deliverable, production path, baseline, target pool settings, primary metrics, decision thresholds, and release acceptance criteria. | Scope and decision rules are written down before new comparisons begin. **Complete 2026-09-25.** |
| 1. Prove the release path | 2 weeks | Trace source data through model, construction, payload, and displayed result. Verify only the data, topology, scoring, provenance, and clean-build contracts on that path. | A clean, documented build reproduces the expected release output; no known release-critical defect remains. |
| 2. Validate release-critical methodology | 2 weeks | Apply the relevant checks from the methodology protocol: leakage and point-in-time integrity, probability/scoring correctness, baseline calibration, and simulation adequacy. | Each required check is PASS, FAIL, or INDETERMINATE with evidence and an explicit release impact. Failures block release; accepted uncertainty is disclosed. |
| 3. Freeze the choice | 1 week | Compare only the predeclared candidate strategy/configuration against the agreed baseline using frozen data, metrics, and decision rules. Retain the baseline if evidence is insufficient. | One configuration is selected with a recorded rationale; no decision-changing analysis remains open. |
| 4. Create the prospective release | Before the 2027 tournament | Freeze code/configuration, data versions, settings, random seeds, and artifact. Record commit, checksums, assumptions, and limitations. | The release artifact and its provenance are reproducible and timestamped. |
| 5. Evaluate the release | After the 2027 tournament | Apply the predeclared evaluation to actual outcomes. Separate the single-season realization from evidence about model quality. | A concise report records results, limitations, and what continues, stops, or remains unknown. |

### Current phase disposition — 2026-09-25

| Phase | Status | Evidence and consequence |
|---|---|---|
| 0. Set the contract | **PASS — complete** | The frozen v4 spec records the deliverable, candidate rules, comparison window, scoring, cutoffs, thresholds, and fallback. |
| 1. Prove the release path | **INDETERMINATE — implementation rehearsed; evidence gate not clear** | Synthetic 2027 payload/build and browser paths pass focused checks. The actual historical-source audit returns FAIL: 13 candidate artifact sidecars mismatch their files and 2012's artifact/sidecar is missing. Separately, the registered public-picks evidence lacks timestamps for 13 seasons and an archive for 2012. No historical comparison ran. A clean committed production build and 2027 payload parity cannot be produced from today's worktree and absent future inputs. |
| 2. Validate release-critical methodology | **INDETERMINATE overall** | ESPN scoring and structural result checks PASS; walk-forward seed-rate cutoff tests PASS. Point-in-time eligibility is INDETERMINATE; seed-only calibration and frozen v4 simulation-ranking adequacy are INDETERMINATE. See the Phase 2 results below. The uncertain items block a candidate superiority claim but do not change the frozen baseline fallback. |
| 3. Freeze the choice | **PASS — seed-only fallback selected** | The frozen decision rule requires seed-only when source or promotion evidence does not PASS. Historical promotion is INDETERMINATE and has no paired score results, so seed-only is the only supported release choice at this checkpoint; this is not a finding that the candidate strategy underperforms. |
| 4. Create the prospective release | **PENDING — not yet actionable** | Selection Sunday is 2027-03-14. The announced field, eligible 2027 Torvik snapshot, and cutoff-compliant public-picks capture do not exist yet. Do not generate or publish a synthetic release artifact. |
| 5. Evaluate the release | **PENDING — after the tournament** | No frozen 2027 release artifact or 2027 tournament outcomes exist. Realized points, pool rank, and post-season report must wait until after the tournament. |

## Phase 0 decisions

**Status: Complete 2026-09-25.** The release target, baseline, selector, scoring, comparison window, thresholds, fallback, source cutoffs, and release evidence requirements are recorded below and in the dated `2027.v4` selection freeze. Phase 0 closes on the frozen contract, not on production execution. The 2027 inputs do not exist yet; source verification, pipeline rehearsal, artifact replay, and site parity remain later phase gates.

### Decisions recorded

- **Release deliverable:** the bracket shown by the production site, frozen with its source data and provenance.
- **Baseline:** a seed-only bracket under the same pool settings.
- **Primary pool setting:** 30 entries, using ESPN standard scoring.
- **Post-season primary outcome:** paired actual ESPN points for the released bracket and seed-only baseline on the same realized 2027 tournament. Actual rank in the real pool leaderboard is secondary.
- **Pre-release comparison:** paired actual ESPN-point differences against seed-only across walk-forward tournament seasons. The candidate must have positive mean gain and a 95% confidence interval whose lower bound is above zero.
- **Historical comparison window:** all 14 eligible completed tournament seasons from 2011–2025, excluding 2020, with strategy selection nested inside each target season.
- **Uncertainty procedure:** primary 95% interval is a paired percentile bootstrap over seasons (5,000 resamples, seed 42); report a paired t-interval as a sensitivity check. Seasons, not games, are the independent units. If the intervals lead to different pass/fail conclusions at zero, classify the result as INDETERMINATE rather than selecting the favorable method.
- **Selection-bias control:** within each target season, select the strategy using only information available before that season. Do not treat a confidence interval on a strategy selected from the same evaluation results as confirmatory evidence.
- **Nested selector:** for each target season, choose among `ev_optimal`, the highest-P(1st) candidate in that season's full bank, and `blend_region_35`. Choose the rule with the greatest mean paired ESPN-point gain over seed-only on earlier eligible target seasons only; an exact tie is broken by rule ID alphabetically. With no earlier eligible target season, use seed-only. For the 2027 release, fit this selector on the fixed 2011–2025 window, excluding 2020.
- **Historical target cutoff rules:** use the announced field and season-specific topology; Torvik must be a pre-tournament snapshot strictly before that season's authoritative tournament start; public picks must be archived before the Round-of-64 lock; the seed referee uses only results with season `< target_year`; First Four results may resolve the target field only before the R64 lock. Target-season tournament outcomes are used only for scoring after the bracket is generated.
- **Historical source-evidence rule:** archives without verifiable capture timestamps are labeled unverified, not assumed timely. Phase 1 must establish source eligibility for every registered fold without changing the fixed target-season list after seeing scores. If the missing provenance prevents a point-in-time claim, the historical promotion result is INDETERMINATE and the already-frozen seed-only fallback applies; do not repair it with post hoc cutoffs or substitute seasons.
- **Fallback:** if the candidate misses the improvement threshold, use the seed-only bracket for the 2027 release and make no superiority claim.
- **INDETERMINATE fallback:** if improvement is not clearly established, also use the seed-only bracket and make no superiority claim.

The single-season post-season result is a realization, not by itself evidence of general model superiority. About 14 historical seasons provide limited independent information; increasing the number of bootstrap resamples does not increase the number of seasons.

**Phase 0 owner:** repository maintainer (sole collaborator). **Start date:** 2026-09-25.

### Phase 0 completion checklist

The decision contract is complete and frozen in [`prospective_2027_v4_points.json`](../configs/frozen/prospective_2027_v4_points.json). The remaining evidence below belongs to the phase gates shown; it is not a reason to leave Phase 0 open.

**Frozen in Phase 0:**

- Deliverable, seed-only comparator, 30-entry ESPN Standard setting, primary/secondary outcomes, exact 14-season target window, nested selector rules, tie breaks, uncertainty method, promotion threshold, and FAIL/INDETERMINATE fallback.
- Point-in-time source rules for historical target seasons and the 2027 prospective season, with the known limitation that some historical archive capture times may not be verifiable.
- Candidate-bank generation command/defaults, missing-result policy, evidence manifest contents, replay procedure, and production-payload parity acceptance criterion.

**Phase 1 must establish before historical comparisons begin:**

- Whether every registered historical fold has eligible, point-in-time source snapshots. The initial inventory is INDETERMINATE: 13 historical public-picks files have no capture timestamps and 2012 has no file; this remains unresolved unless independent pre-tip evidence is found.
- The exact baseline construction, score implementation, and repeatable fold-generation/selector path. Result completeness and integrity have now been checked for available seasons and adversarial cases.
- Any source-integrity or release-path defect that could invalidate the comparison; record its bounded impact and exit decision.

### Phase 1 work-item result: historical source eligibility — INDETERMINATE

Inventory covered the frozen 14 targets: 2011–2019 and 2021–2025. The exact readers and checks are `scripts/experiments/build_candidate_artifact.py` (`assert_pretournament_inputs`, `_torvik_provenance`, `_public_picks_provenance`, `resolve_field`), `src/data/seed_pick_model.py` (`_recent_win_rates(as_of)`), `src/data/historical_picks.py` (`archive_candidates`), and the season-specific topology resolver in `src/simulation/bracket_topology.py`.

| Input contract | Observed status | Decision |
|---|---|---|
| Announced-field inputs and played-in field | All 14 context files contain 68 entered teams; four recorded First Four games resolve each field to 64. Both played Final Four games exist in each results file and yield a unique four-region pairing. | PASS for structural availability and field/topology resolution. Historical topology is reconstructed from the played F4 matchups; it is not independent capture-time evidence of the announced bracket. |
| Tournament results | All 14 files contain the expected 67 games: 4 FF + 63 scored games, with round counts 32/16/8/4/2/1 from R64 through NCG. | PASS for counts only. Duplicate-game, participant-chain, and winner-integrity checks remain a separate Phase 1 work item. |
| Torvik | All 14 snapshots are labeled `pre_tournament`; each has a cutoff before its authoritative tournament-start date and matching `tournament_start`. | PASS for the recorded date boundary and metadata consistency. |
| ESPN public picks | Archives exist for 13 targets; every one lacks `captured_at`/`timestamp` and carries source attribution but no capture-time proof. No archive exists for 2012; the candidate builder explicitly refuses that season because the P(1st) candidate bank requires these picks. | FAIL to establish complete point-in-time eligibility. Do not infer capture dates from current file times or source attribution; do not synthesize a timestamp or remove 2012 from the frozen target window. |
| Seed referee | Both Kaggle seed/results files exist. The candidate path calls the seed probability builder with `as_of=target_year`; `_recent_win_rates` filters out every result season `>= as_of`. | PASS for walk-forward season filtering in the inspected code path. Input snapshot hashes are recorded at candidate-build time. |

**Overall result: INDETERMINATE for the 14-season historical promotion comparison.** Thirteen folds lack verifiable public-picks timestamps and the 2012 public-picks input is absent, so exact candidate-bank regeneration and the frozen point-in-time claim cannot be established for the complete fixed window. This does not change the target set and no ESPN-points comparison was run. Under the frozen rule, no historical superiority claim is supportable on this evidence; retain seed-only unless independent pre-tip evidence is recovered without changing the preregistered seasons or procedure. Revisit only if new material provenance evidence appears.

### Phase 1 work-item result: result completeness and advancement integrity — PASS

The track-record path previously required nonempty round winner sets and one champion, allowing partial results to understate points. Its guard now runs before winner extraction and rejects incorrect game counts, unknown rounds, missing/wrong season labels, non-object rows, duplicate matchups, repeat participants in a round, invalid/equal team IDs, non-boolean winner flags, missing/non-integer/tied scores, winners inconsistent with scores, incorrect First Four inclusion in the Round of 64, broken participant-to-prior-winner chains, and incorrect round winner counts. Independent review caught and closed three edge cases: non-object rows crashing before validation, missing years passing, and boolean scores passing as integers. All 14 fixed target-season result files plus 2026 passed the structural guard; focused adversarial regression tests now cover these defects as well as incomplete rounds, duplicate matchups, and broken advancement.

The guard validates the shape and internal consistency of the results source, not the external correctness of reported game outcomes or the fidelity of historical picks. It does not resolve the separate public-picks source-evidence indeterminacy.

### Phase 2 results: release-critical methodology

Bounded checks use only the release-critical areas named in Phase 2; this is not a full foundational audit.

| Check | Status | Evidence and release impact |
|---|---|---|
| Walk-forward seed-rate leakage guards | **PASS for the checked code paths; overall point-in-time gate INDETERMINATE** | `tests/test_seed_table_walk_forward.py`, `tests/test_noseed_round_probabilities_walkforward.py`, `tests/test_leakage_canary.py`, and `tests/test_pit_production_port.py` test strict target-season exclusion and production-port consistency. They cannot establish the missing historical public-picks capture times or the absent 2012 archive. |
| ESPN Standard scoring and result structure | **PASS for implementation and available structural fixtures** | `tests/test_scoring_independent.py` compares the scorer with an independent slot-wise reference; the complete-outcome guard validates registered available results and adversarial malformed-result cases. This does not independently verify the external truth of each historical box score. |
| Seed-only baseline | **PASS for deterministic construction/fallback behavior; INDETERMINATE for calibration** | Selector tests verify target-year `as_of`, no public-picks input, and fallback selection. Current evidence does not provide a held-out calibration result for the seed-only bracket. The blocked historical candidate comparison is not replaced with an unregistered baseline experiment. |
| Simulation adequacy | **INDETERMINATE for frozen v4 ranking** | Deterministic common-random-number tests and a 20,000-simulation toy convergence test pass, but there is no evidence that 150,000 generated tournaments and 2,000 P(1st) trials stabilize the ordering of the frozen v4 rules. The existing Phase 2 integration fixture is v2/2026 and explicitly not a v4 evaluation; it is not accepted as evidence. |

Validation:

```text
pytest -q tests/test_seed_table_walk_forward.py tests/test_noseed_round_probabilities_walkforward.py tests/test_leakage_canary.py tests/test_pit_production_port.py tests/test_scoring_independent.py tests/test_tournament_results_integrity.py tests/test_selection_common_random_numbers.py tests/test_prospective_2027_points_freeze.py tests/test_backtest_scoring_and_opponent_defaults.py
62 passed

pytest -q tests/test_objective_exhaustive.py tests/test_tournament_results_integrity.py tests/test_prospective_2027_points_selector.py
45 passed
```

The separate end-to-end selector/source/rehearsal set passed 97 tests; browser behavior passed 66 checks. Overall Phase 2 remains INDETERMINATE; no v4 superiority claim is authorized. Do not expand into unrelated methodology checks unless new release-critical evidence changes the decision.

**Phase 4 must establish before publishing the 2027 bracket:**

- Actual captured source paths, timestamps/cutoffs, code/runtime versions, settings, and hashes.
- Frozen candidate and seed-only brackets, source manifest, immutable artifact SHA-256, exact replay, and production-site payload equality by team identity in every round.

The [2027 season payload](./data/season_2027.json) is currently `not_started`; therefore 2027 source hashes, artifact replay, and live payload parity cannot yet be measured. This is an explicit future acceptance gate, not a Phase 0 omission.

## 2027.v4 selection re-freeze — 2026-09-25

Machine-readable freeze: [`configs/frozen/prospective_2027_v4_points.json`](../configs/frozen/prospective_2027_v4_points.json), SHA-256 `797aba3e8db5a2b23cecbc0efa359445255bf5430f872d5b4eb393089bd28ee3`.

This dated freeze supersedes [the immutable 2027.v3 methodology spec](../configs/frozen/prospective_2027_v3_audited.json) for **which bracket is selected for release and the historical promotion gate**. It retains v3's point-in-time model inputs and candidate-bank generation as the candidate-generation method; v3 remains unchanged as a historical record. Do not revise either freeze after evaluating the registered seasons.

### Candidate bank and selector

- Candidate universe: every generated and constructed candidate in the season's `candidates_YEAR.json` bank, including its named strategies.
- Candidate-generation defaults: 150,000 total simulated tournaments, 3,000 target candidates, 2,000 P(1st) trials, RNG seed `20260820`, pool size 30, `espn_standard`. The simulation total is divided evenly across whichever rating sources are available; Torvik is mandatory, while Massey and Elo are currently optional and their availability must be recorded. Missing Torvik team ratings currently use the existing seed-based estimate; enumerate those teams in the release record rather than treating them as measured Torvik values.
- Registered rules: `ev_optimal` (the artifact's named exact expected-points bracket); `highest_p1` (the first index returned by `select_diverse(artifact, objective="p1", k=1)`, which is the highest P(1st) candidate and resolves ties by artifact order); and `blend_region_35` (the artifact's named fixed blend bracket).
- Inner selection: score each registered rule on each earlier eligible target season against that season's seed-only bracket. Select the rule with the highest mean paired point gain. The first outer target, 2011, has no earlier eligible target season and therefore uses seed-only. The final 2027 rule selection uses exactly the registered 2011–2025 seasons excluding 2020; do not add 2026 after seeing its result.
- Outer promotion gate: the 14 walk-forward target-season differences must have positive mean and a paired 95% season-bootstrap percentile interval (5,000 resamples, seed 42) whose lower bound is above zero. Report the paired t-interval as a sensitivity check; if it changes the pass/fail conclusion at zero, classify the outcome as INDETERMINATE. FAIL or INDETERMINATE means seed-only for 2027 and no superiority claim.

### Point-in-time inputs and cutoffs

| Input | Frozen rule for 2027 | What must be captured at build time |
|---|---|---|
| Announced field and topology | Selection Sunday is **2027-03-14**. Use the announced 64-team field and explicit `f4_pairing`; resolve First Four slots only from recorded First Four results before the R64 bracket locks. | Exact source file/path, source or publication timestamp, `f4_pairing`, First Four result rows used, and SHA-256. |
| Torvik | `data_type` must be `pre_tournament`; `cutoff_date` must be strictly before the authoritative tournament start **2027-03-16** (thus the latest eligible date is **2027-03-15**). The declared `tournament_start` must match the season calendar. | Actual cutoff date, source file/path, file SHA-256, and the rating-source coverage/fallback list. The builder now enforces this date boundary. |
| ESPN public picks | One archived capture, with timezone-aware `captured_at`/`timestamp`, at or before **2027-03-18 12:00 America/New_York**. No recapture after the cutoff; a missing or late timestamp blocks the official build. | Archive path, source, exact capture instant, and SHA-256. |
| Seed referee | `seed_pick_model` recent seed-vs-seed rates with `as_of=2027`; results from 2027 or later must not enter. | Kaggle seed/results file paths and hashes, and the resolved `as_of` value. |
| Other candidate-bank inputs | Use the source snapshots actually consumed by the frozen generator; optional Massey/Elo absence narrows the bank and must be recorded, not silently substituted. | File paths, timestamps/cutoffs, SHA-256s, and source availability. |

All score comparisons use ESPN Standard team-identity scoring: R64=10, R32=20, S16=40, E8=80, F4=160, championship=320; First Four games score zero. The frozen contract requires a complete tournament result: 4 FF, 32 R64, 16 R32, 8 S16, 4 E8, 2 F4, and 1 NCG game, with winner-set sizes 32/16/8/4/2/1 for R64 through champion. Missing, duplicated, malformed, or incomplete round results produce **no score**; do not impute missing games or treat them as losses/zero points. The `scripts/build_track_record.py` guard checks exact per-round game counts and winner-set sizes, and rejects duplicate games and broken advancement links before scoring.

### Build, evidence, and release acceptance

Canonical production invocation (defaults are part of this freeze):

```bash
python -m scripts.build_season --year 2027
```

The command refreshes the team-stat payload, builds `artifacts/candidates/candidates_2027.json`, and rebuilds `docs/data/season_*.json` plus `docs/data/seasons.json`. The official candidate artifact is immutable: the builder must fail rather than overwrite it. Review the validation output before publishing.

For each historical fold, regenerate that season's bank with the same candidate settings and point-in-time inputs, then apply only the registered inner selector. The baseline is the season's seed-only bracket (the lower-numbered seed advances, using the same resolved field and bracket topology). Paired ESPN points are computed from the same complete actual-results source with the canonical team-identity scorer. Historical picks lacking capture timestamps remain unverified; if the Phase 1 provenance check cannot establish fold eligibility, the fixed comparison is INDETERMINATE rather than silently excluding those seasons.

The release evidence record must contain:

- Freeze version/hash; clean Git commit SHA; `git status --porcelain` empty; Python and dependency versions (or lockfile hash).
- Every consumed source file's role, repository-relative path, source/capture/cutoff date, and SHA-256, including context/field, Torvik, archived picks, Kaggle seed/results, and each available optional rating source.
- Exact command, all settings and RNG seeds, artifact `generated_at`, candidate artifact SHA-256 and matching `.sha256` sidecar, seed-only bracket SHA-256, frozen site-payload SHA-256, and relevant output hashes.
- The selected rule ID and its rule-selection table; the 14 paired point differences, bootstrap interval, t-interval sensitivity, and PASS/FAIL/INDETERMINATE decision.
- Replay evidence: regenerate from the same commit, source snapshots, settings, and timestamp into a separate output directory; verify the candidate artifact checksum and bracket; verify the production payload's selected bracket by resolving team IDs and comparing every round to that frozen artifact.

The candidate artifact now embeds hashes for Torvik and public-pick inputs, field/results inputs, Kaggle seed-referee files, and available optional rating-source files. It also records the RNG seed, candidate target, per-source simulation count, source set, and any missing Torvik team ratings. The external release record supplies the clean code revision, runtime/dependency versions, and hashes of the candidate, baseline, and production payload.

For a byte-for-byte candidate-artifact replay, pass the recorded UTC `generated_at` to `scripts.experiments.build_candidate_artifact --generated-at`; the option accepts only timezone-aware ISO timestamps. Build the replay into a separate output directory so the official artifact's overwrite protection remains intact.

**Implementation status:** the frozen `2027.v4` nested selector, seed-only baseline, paired-score gates, source-evidence checks, CLI, and production `Recommended` strategy are implemented and wired. The browser defaults to Recommended while retaining P(1st), EV, and champion/filter choices. A selector result is usable only when its report hash and current inputs verify; otherwise Recommended explicitly falls back to seed-only and supplies no invented pool scores.

**Current release blockers:** the historical evaluator was run and returned `FAIL` at its source-integrity gate: 13 candidate artifact SHA-256 sidecars disagree with their artifacts, and the 2012 candidate artifact and sidecar are missing. Independently, the registered historical public-picks archives still lack verified capture timestamps for 13 seasons, and 2012 has no archive. No paired score comparison or promotion decision was made; promotion remains `INDETERMINATE`, and the frozen seed-only fallback applies. Do not repair hashes or infer timestamps to force eligibility. The [2027 season payload](./data/season_2027.json) remains a `not_started` placeholder; the 2027 field, Torvik snapshot, and timestamped pick archive are absent, so no 2027 artifact replay or production-payload parity evidence exists yet. The full `pytest -q` run is not green: it first stops at the existing source-string assertion `tests/test_fitted_eval.py::test_evaluator_only_reads_the_candidate_artifact`; excluding that test next fails the existing pool-size assertion `test_configuration_matches_the_artifact[fitted_eval_2011_pool100_espn_standard]`. These do not change the source-gate result and are not modified in this focused release task.

Run-specific source timestamps, commits, hashes, output artifact, selector result, and payload parity are **pending** because the required 2027 inputs do not yet exist. Do not claim the prospective artifact is built or reproducible until those values are captured and the replay/parity checks pass.

## Work-item contract

Before starting an investigation, add a short entry to the decision log containing:

1. **Question:** What decision will this answer?
2. **Release relevance:** Which phase gate could it affect, and how?
3. **Method:** What data, comparison, and measure will be used?
4. **Timebox and exit criterion:** When does work stop, and what constitutes a result?

At completion, record the result, evidence location, decision, and any downstream impact. If no decision changes, do not continue analysis merely to produce more detail.

## Decision log

| Date | Question / decision | Evidence | Outcome and impact |
|---|---|---|---|
| 2026-09-25 | Set the program finish line | Session planning discussion | Agreed: reproducible prospective 2027 release plus preplanned post-tournament evaluation. |
| 2026-09-25 | Define the prospective deliverable | User decision | Freeze the production-site bracket with source data and provenance. |
| 2026-09-25 | Select baseline and pool contract | User decision and supported settings in `docs/OPERATIONS.md` | Compare to a seed-only bracket at 30 entries and ESPN standard scoring. |
| 2026-09-25 | Set post-season outcomes | User decision | Paired actual ESPN points are primary; actual pool leaderboard rank is secondary. A single tournament is descriptive, not general proof. |
| 2026-09-25 | Set pre-release comparison measure | User decision | Paired actual ESPN-point differences versus seed-only across walk-forward seasons; report uncertainty across seasons and set a numeric release threshold before analysis. |
| 2026-09-25 | Set candidate improvement threshold | User decision | Require positive mean ESPN-point gain and a 95% confidence interval with lower bound above zero. |
| 2026-09-25 | Define fallback if threshold is missed | User decision | Use seed-only bracket for the 2027 release and make no superiority claim. |
| 2026-09-25 | Predeclare uncertainty and selection-bias controls | User accepted recommended rule | Primary paired percentile season bootstrap (5,000 resamples, seed 42); paired t-interval sensitivity; differing pass/fail conclusions at zero are INDETERMINATE. Strategy selection is nested within each target season. |
| 2026-09-25 | Define INDETERMINATE fallback | User decision | Use seed-only bracket and make no superiority claim whenever improvement is not clearly established. |
| 2026-09-25 | Fix historical comparison window | User decision | Use all 14 eligible seasons from 2011–2025 excluding 2020, with candidate/strategy selection nested within each target season. |
| 2026-09-25 | Supersede v3 production selection rule | User decision | Create a dated re-freeze using a nested ESPN-points selector over the full season candidate bank; v3 remains immutable and supplies candidate-generation inputs. |
| 2026-09-25 | Fix selector candidate rules | User decision and existing artifact contract | Compare `ev_optimal`, highest-P(1st) bank candidate, and `blend_region_35`; seed-only remains comparator/fallback. |
| 2026-09-25 | Define nested rule selection | User decision | Select by mean prior-season paired ESPN-point gain, use only earlier eligible target seasons, tie-break alphabetically by rule ID, and use seed-only for 2011's empty training history. Fit 2027 using only the fixed 2011–2025 eligible window. |
| 2026-09-25 | Close Phase 0 at the frozen-contract boundary | User decision | Phase 0 is complete when the contract is defined and versioned. Historical source verification is a Phase 1 gate; actual 2027 source capture, artifact replay, and production-payload parity are Phase 4 acceptance gates. |
| 2026-09-25 | Enforce the Torvik date cutoff | Release-path verification | The candidate build checked `data_type` but did not enforce `cutoff_date`; it now requires the cutoff to be strictly before the authoritative tournament start and verifies the declared start date. |
| 2026-09-25 | Reject incomplete scored outcomes | Release-path verification | Track-record generation now requires exact game and winner counts before scoring; incomplete outcomes produce an explicit error, not an understated score. |
| 2026-09-25 | Phase 1 work item: historical source eligibility | Question: Do all 14 frozen target seasons have the required point-in-time source inputs, and can their temporal eligibility be evidenced? Relevance: validity of the registered historical promotion gate. Method: inventory actual field/results, Torvik, public-picks, and Kaggle seed-referee inputs and loader guards without scoring candidates. | **INDETERMINATE.** All fields resolve 68→64, all 14 Torvik cutoffs pass, and Kaggle seed rates are filtered with `as_of=target_year`; but 13/14 public-picks archives have no capture timestamp and 2012 has no archive. Fixed target window retained; no comparison run; seed-only fallback remains in force absent independent timestamp evidence. |
| 2026-09-25 | Phase 1 work item: result completeness and advancement integrity | Question: Can malformed, duplicated, or mislinked tournament result rows pass the frozen complete-outcome guard and receive an ESPN score? Relevance: Phase 1 scoring correctness and validity of every paired target-season outcome. Method: inspect actual-results extraction, track-record validation, and caller tests; use adversarial fixtures for duplicate games, invalid winners, and broken round-to-round participants. | **PASS for structural integrity.** Strengthened the pre-scoring guard and added adversarial regression tests; all 15 available result files (2011–2026 excluding 2020) pass. Validation: focused tests 4 passed; Ruff passed. |
| 2026-09-25 | Phase 1 work item: freeze and wire v4 selection | Question: Can the frozen walk-forward selector, point-in-time source gate, seed-only fallback, and Recommended site strategy run end to end without presenting unverified historical or pool metrics? Relevance: Phase 1 production and evaluation path. Method: implement the frozen selector/score gates, exercise the expanded-field 2027 payload rehearsal, test browser default/share/export, and run the repository source gate. | **Implementation and wiring PASS; historical evidence gate FAIL.** Focused selector/freeze/provenance/2027-rehearsal tests: 97 passed; browser checks: 66 passed; pending/stale-payload E2E: 4 passed; Ruff and asset-stamp checks passed. Track-record artifact parity cases: 52 passed. The full track-record module still has a pre-existing 2026 fitted-evaluation timestamp mismatch (and the result-path audit test has a legacy artifact-name parsing failure). Historical audit returned source `FAIL`, promotion `INDETERMINATE`, selected `seed_only`; no paired score comparison ran. 2027 inputs and replay/parity evidence remain pending. |
| 2026-09-25 | Phase 2 bounded release-critical checks | Question: Do the preregistered source timing, scoring, baseline, and simulation checks permit candidate promotion? Relevance: Phase 2 gate and Phase 3 rule decision. Method: targeted leakage/point-in-time tests, independent scoring reference, result-integrity validation, selector baseline/fallback tests, common-random-number tests, and existing v4 simulation-setting evidence only. | **INDETERMINATE overall.** Scoring/available result structure and tested walk-forward implementation pass; complete historical point-in-time eligibility, seed-only calibration, and v4 rule-ranking stability remain unestablished. No simulation adequacy result was inferred from the v2 2026 fixture or toy convergence test. No candidate superiority claim. |
| 2026-09-25 | Phase 3 frozen choice at current evidence boundary | Question: Which 2027 choice is justified when source and promotion evidence are not PASS? Relevance: freeze the release strategy without changing the v4 rule. Method: apply the already-frozen FAIL/INDETERMINATE fallback to the actual source-gate result; do not score or substitute seasons. | **PASS: seed-only fallback selected for the current checkpoint.** This is the contractually required conservative choice, not an empirical claim that the v4 candidates lose. Revisit only if complete, trustworthy registered-fold evidence becomes available before release. |

## Parking lot

These are not release prerequisites unless a work item demonstrates a direct effect on an active gate:

- Broad strategy and hyperparameter searches beyond the predeclared comparison
- Re-analysis of historical experiments that do not support the production path
- UI redesign and feature expansion
- Reopening completed methodology decisions without new material evidence

Add an item only with a short reason it may matter. Triage the list at the end of the release cycle, not during every active phase.

## Closing the cycle

The post-season report closes this roadmap's cycle. New work should start from its conclusions and a newly agreed scope; do not keep extending the 2027 roadmap with unbounded follow-up phases.
