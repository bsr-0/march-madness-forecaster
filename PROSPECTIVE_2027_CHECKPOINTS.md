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

**Status:** OPEN. No decision has been made. Not to be decided by an agent.

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

**Status:** OPEN, pending 2027 data existing at all.

---

## CHECKPOINT 3 — Whether v1 survives the 2026 integration test

**Decide after:** the 2026 integration pass.

The 2026 run is a correctness test, not an evaluation. But if it surfaces a
genuine defect that can only be fixed by changing a frozen parameter, that is a
v2 event and must be reported as one rather than absorbed.

The distinction to hold: a bug in the *implementation* of a frozen parameter can
be fixed under v1 (the parameter's value is unchanged). A change to the
parameter's *value or definition* cannot.

**Status:** OPEN, pending the integration pass.
