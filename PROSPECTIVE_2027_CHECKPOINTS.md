# 2027 Prospective Checkpoints

Companion to `PROSPECTIVE_2027.md`. That document is the frozen contract and its
substance does not change. This one records decisions that are **deliberately
deferred** to a stated point in time, so that deferring them is a choice on the
record rather than an omission discovered later.

Nothing here may alter `configs/frozen/prospective_2027.json` without a
`SPEC_VERSION` bump. The CI gate in `tests/test_frozen_2027_spec.py` enforces that.

---

## Three cutoffs, deliberately distinct

The freeze conflates two of these if you are not careful, and they answer
different questions:

| | what it governs | status |
|---|---|---|
| **Methodology cutoff** | how the system works — architecture, objectives, sampler, diversity, preferences | **FROZEN** at 2027.v1, 2026-08-20 |
| **Training-data cutoff** | which historical seasons the model may learn from | currently **2025**; open until the checkpoint below |
| **Prediction-time cutoff** | what is knowable when a given bracket is generated | per artifact; recorded in each artifact's `provenance` |

Keeping them separate is what allows the training data to be legitimately
refreshed before March 2027 without weakening the prospective claim about
*methodology*, and what stops "we updated the data" from quietly becoming "we
changed the system after seeing results."

---

## CHECKPOINT 1 — Training-data cutoff

**Decide before:** 2027 Selection Sunday, and before any 2027 candidate artifact
is generated.

**Question.** `TRAIN_YEARS` currently ends at **2025**. The 2026 season concluded
in April 2026, months before this freeze, and would legitimately have been
available to anyone building a 2027 bracket. Should it enter training?

**Why this was not settled at freeze time.** Extending training data is a model
change, and making it silently would have undermined the freeze it was made
under. It is recorded in `PROSPECTIVE_2027.md` as a known gap for exactly this
reason.

**What makes a decision legitimate.** The decision must be made *before* any 2027
outcome exists, and must not be justified by any 2027 result. It is a judgement
about what information a user would have had, not about what improves a score.

**The argument for including 2026 (extend to ≤2026)**

- it is genuinely pre-2027 information; excluding it makes the system weaker than
  it needs to be, for no methodological reason
- a real user building a 2027 bracket would obviously have known 2026's results
- the training window is otherwise 18 seasons; one more is a marginal, principled
  addition rather than a tuning lever

**The argument for retaining 2025**

- 2026 has already influenced this project's research decisions, so folding it
  into training mixes an in-sample season into the model behind a prospective
  claim
- retaining 2025 keeps v1 exactly as frozen, with no retraining or revalidation
- the marginal accuracy gain from one season is likely small relative to the
  interpretive cost

**Consequence either way.** Choosing to extend is **v2**: bump `SPEC_VERSION`,
regenerate the frozen spec, retrain, revalidate, and record in a new
`PROSPECTIVE_2027_v2.md` that the v1 prospective claim is superseded. Choosing to
retain requires nothing — v1 stands as frozen.

**Note.** Whichever is chosen, the 2026 *outcome contamination* argument in
`PROSPECTIVE_2027.md` still stands: 2026 cannot be an evaluation season either
way. This checkpoint is only about whether the model may *learn* from it.

**Status:** CLOSED, 2026-08-20 — extend to ≤2026. Recorded in
`PROSPECTIVE_2027_v2.md`; `TRAIN_YEARS` in `src/prediction/noseed_model.py`
carries the 2026 entry and the rationale. This status line previously still read
OPEN after the decision had been taken, which is the sort of stale record that
makes a freeze less trustworthy than it is.

---

## CHECKPOINT 2 — Prediction-time cutoff for the official 2027 artifact

**Decide before:** generating the official 2027 prediction artifact.

Every input needs a stated availability timestamp: seeds (Selection Sunday),
Torvik ratings (last pre-tournament snapshot), public pick percentages (as close
to tip as is genuinely available). The artifact records these per input, and
`assert_pretournament_inputs` refuses to build if any source fails its provenance
check.

The rule that matters: **once the official artifact is generated it is
immutable.** Regenerating it after later information and presenting the result as
the original prediction is the single failure mode that would void the entire
exercise.

**Decided 2026-09-05, before any 2027 information existed.**

| input | cutoff | enforced by |
|---|---|---|
| seeds | bracket release, Selection Sunday 2027-03-14 | `load_seeds_and_regions` |
| Torvik ratings | last `trank.php` snapshot with `data_type == "pre_tournament"` | `assert_pretournament_inputs` |
| public pick shares | **2027-03-18 12:00 ET**, one capture, no re-capture | `_public_picks_provenance` |

**Why the picks cutoff is a fixed instant rather than "as late as available."**
Pick shares move until tip, so "as late as available" has no failure condition:
it cannot distinguish a legitimate late capture from a re-capture taken after
seeing something. A named instant, chosen before the season, can be met or
missed. 2027-03-18 12:00 ET is roughly fifteen minutes before the first R64 tip
— late enough that the field has committed, early enough to be a deadline rather
than a race. It is declared in `season_calendar.PUBLIC_PICKS_CUTOFFS` and pinned
by a test that says changing it must be deliberate.

**What was implemented to make this enforceable rather than aspirational.** The
gate previously verified Torvik and described the seed head-to-head table, and
said nothing at all about public picks — the one input this checkpoint singles
out. It now requires a timezone-aware capture time for any season with a
declared cutoff, refuses a capture past that cutoff, and refuses any capture
past the R64 tip for every season. Seasons without a declared cutoff are
historical validation reading archives whose capture time nobody recorded; a
missing timestamp is tolerated there and recorded in the artifact as
`capture_time_verified: false` rather than passed over.

Immutability now has a mechanism too: `main` writes a `.sha256` beside the
artifact and refuses to overwrite an existing one for a declared season, with or
without `--force`. Deleting the file by hand still works, which is the point —
the destructive act should be separate and deliberate, not a side effect of
re-running a build command.

**Status:** CLOSED for the decision and its enforcement. The capture itself
remains to be performed in March 2027.

---

## CHECKPOINT 3 — Whether v1 survives the 2026 integration test

**Decide after:** the 2026 integration pass.

The 2026 run is a correctness test, not an evaluation. But if it surfaces a
genuine defect that can only be fixed by changing a frozen parameter, that is a
v2 event and must be reported as one rather than absorbed.

The distinction to hold: a bug in the *implementation* of a frozen parameter can
be fixed under v1 (the parameter's value is unchanged). A change to the
parameter's *value or definition* cannot.

**Run 2026-09-05: 30/32, then 32/32 after two fixes. Neither fix touched a
frozen parameter, so v2 stands and this is NOT a v3 event.**

Both failures were defects in the *checks*, not in the system, and both are
worth stating rather than filing as green:

**1. "system matches the 2027.v1 freeze — 9 drifted."** The test named
`configs/frozen/prospective_2027.json` explicitly, overriding
`diff_against_frozen`'s default of `FROZEN_SPEC_PATH` — which is the operative
`prospective_2027_v2_scoped.json` that `tests/test_frozen_2027_spec.py` actually
gates on. So it was measuring the live system against the **superseded** v1
spec. All 9 drifted fields were the documented v2 changes: `train_years` and
`training_cutoff_season` extended to 2026, the three fields reclassified to
`product.v3`, and v2's own bookkeeping (`spec_version`, `supersedes`,
`scope_correction`, `holdout.contaminated_for_evaluation_only`). Against the
operative spec the live system drifts on **zero** fields. It was exactly as
frozen the whole time; the test was pinned to the wrong freeze.

**2. "EV recomputes exactly — max err 1.14e+02."** `validate` re-derived
marginals from whatever rounds list it was passed, and the call site passed
every candidate from all three rating sources — while EV is defined against
Torvik's marginals alone, deliberately, so scores stay comparable. The check
compared a Torvik EV to a three-source EV and reported the gap between two
rating systems as an arithmetic error. It now takes the marginals rather than
re-deriving them, and the error is exactly 0.0.

This one was failing in the shipped artifacts, not just in the test: every
`candidates_*.json` built since the pool was broadened carries
`ev_max_abs_error` between 81 and 142. All artifacts were rebuilt after the fix
and now record 0.0; the rebuilt candidates are otherwise byte-identical, so no
UI payload changed.

**Both checks passed when they were written.** The record committed at
`aa4f104` shows 32/32 with `max err 0.00e+00`, from a time when there was one
rating source and v1 was operative. Two ordinary changes -- `ae617ce`
broadening the pool to three sources, and the v2 freeze -- invalidated them,
and the test is referenced by no workflow and no test file, so nothing re-ran
it. **This is the thing to fix if the pass is to mean anything in 2027:**
either wire `integration_test_2026.py` into CI or treat its recorded result as
describing whatever system existed when someone last typed its name.

**Status:** CLOSED. v2 survives the integration pass. The two defects were in
the verification layer, which is its own small lesson: the checks are code too,
and 32/32 on a check that measures the wrong thing is worth less than a
failure that means something.
