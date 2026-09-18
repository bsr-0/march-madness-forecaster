# Rule search — experimental strategy (rewritten 2026-09-18)

Not a methodology change. A fenced, client-side feature on the site; nothing
in production scoring, selection, the candidate artifact, the fitted
evaluation or the track record reads it. Never a P(1st), EV, accuracy or
ranking.

## What it asks
*How far back can one rule reproduce these rounds?* A rule is one criterion
per round: in that round every game goes to the team better on one
variable (direction-corrected z; ties to the better seed, then lower index,
the board's own rule). Two controls and no others:

- **Reproduce through** — any of Sweet 16, Elite Eight, Final Four,
  Champion (rounds 1, 2, 3, 5). Default Final Four + Champion. At least one
  of the first three must stay chosen: with only the champion nothing
  prunes before the last round (keys^5 sequences per season); the panel
  says so when a click would leave that.
- **Rule complexity** — Simple (at most two distinct variables across the
  rounds; default) or Flexible (at most three). No uncapped search is
  offered; see the measurements.

The window the rule is asked to reproduce is not a control. The eligible
criteria are every variable plus seed, always.

## The automatic window (`ruleRun`, fit.js)
Played seasons strictly before the displayed one, newest first. *Skip
phase:* a newest season that no rule reproduces on its own is recorded and
passed over — the run has to start at the most recent season some rule
can reproduce, and the panel names the seasons it skipped. *Run phase:*
from that season, the window grows one season back while the intersection
of survivors is non-empty; the season that leaves none is `stoppedAt`.
Seasons past the break are never enumerated. Survivors are ranked fewest
`RULE_GENERAL_KEYS` criteria first — barthag, national rank, Massey rank,
strength of schedule, Simple rating, adjusted offense, adjusted defense,
and seed: composite ratings and rank-of-ratings, real and correlated with
winning, but a whole season folded into one number rather than the one
specific thing the search is meant to surface. Tempo and every box-score,
form and roster variable count as specific and rank ahead of them when a
rule of the same complexity survives on either side. *Then* simplest
(distinct criteria, switches, criterion order round by round). This only
reorders the survivors; a rule leaning on a rating is still offered when
no specific-only rule reproduces the window (see `ruleGeneralCount`, and
`general` on each offered entry). The cap is applied inside the
enumeration (`ruleSequencesForSeason` carries a criterion mask per
sequence), which is what makes both settings cheap; the set is identical
to filtering afterwards, and the specific-first ranking doesn't touch it.

## Three things, kept apart
| On the panel | Meaning | Selected on? |
|---|---|---|
| **Recent run** | The consecutive seasons, newest back, that the rule reproduces | Yes — this is what the search found |
| **Other matches** and the **per-round record** | Older seasons outside the run (skipped ones included) the chosen rule also reproduces; and, per chosen round, how many of all played seasons it reproduces | No — descriptive; never extends the run |
| **This season's bracket** | The rule applied to the displayed field; under a pending field, the criteria under empty round headers | — |

The simplest survivor is offered with two alternatives (`RULE_OFFERED`,
distinct by the bracket they give when a field exists, by sequence when
not). Beside them: how many survivors reproduce the rounds in at least one
older season (`ruleGeneralisation`, exact within 200,000 applications, else
on the simplest prefix with the copy saying so). When the run is at most
two seasons and fewer than half the checked survivors repeat, that is
shown as the **short-history warning** rather than as a survivor count:
the count is a diagnostic, the generalisation is the information.

## Measured on the shipped data (2026-09-18, Final Four + Champion, 32 criteria)
| Cap | 2027 page (played 2011–2026) | 2026 page | Cost |
|---|---|---|---|
| Simple (≤2) | run **2026**; 2025 breaks it; 72 rules, none repeat in the 13 older seasons | run 2025; 86 rules | ~40 ms |
| Flexible (≤3) | run **2025–2026**; 2024 breaks it; 39 rules, none repeat | run 2025; 12,297 rules | ~0.2–0.4 s |
| ≤4 / uncapped | the same runs | 293k / **5.25M** rules, ~600 MB | seconds |

Flexible is capped at three because the uncapped search reaches no longer
run at 400× the survivors and a memory cost a phone would not survive. On
the **2025** page no rule reproduces the Final Four in 2024, 2023, 2022 or
2021 at either cap (the 4-Alabama / 11-NC State Final Four of 2024 has no
one-variable-per-round reproduction even for the Sweet 16 alone); Simple
starts its run at 2019 and says so. The counts above are unchanged by the
specific-first ranking (it reorders survivors, never prunes them); which
rule is *primary* is what moved. The primary 2027 rule under Simple is now
Shot defense → Shot defense → Shooting (eFG%) → Shot defense → Shot
defense → Shot defense (0 rating variables; before this ranking existed
the simplest-by-code survivor was Tempo → Simple rating ×5, which used one).
It reproduces the Final Four in 1 of 14 played seasons and the champion in
2 of 14 — the same record, since both are drawn from the same 72
survivors of the same search; only the offered order differs.

## The chosen rule, and links
The chosen rule travels in the URL as its criterion sequence (`rq`), with
the rounds (`rc`) and `rs=flexible` when set. It is resolved against the
survivors: by sequence; then, with a field, by the bracket it gives (the
entry then *is* the named rule, with its own numbers); a survivor not among
the offered three is listed as one more; a named rule that did not survive
is said on the panel (`wantMissed`) and kept in the link, since another
season may have it. A season change keeps the choice; a control change
resets it. Carrying a 2027 Simple rule to 2026 is normally `wantMissed`: no
Simple rule reproduces both 2026 and 2025.

## A season whose field is not out (2027 today)
The newest listed season opens by default. The strategy table lists Win
the pool, Most expected points and the fitted model as awaiting the field
(not selectable); the same search runs over 2026, 2025, … (2026 is a prior
season from 2027's side, which is why it belongs in 2027's window and never
in its own); entries have no picks, `ruleStrategy()` resolves to nothing,
and the board is six round headers with the chosen rule's criteria and the
game counts. Once the field ships, the same link fills the bracket with
that rule. The orientation line is the first thing on the page.

## Off the main thread
The whole job is one pure function, `ruleSearchJob()` in fit.js, run in a
blob Web Worker importing fit.js by its stamped URL (`runRuleSearchJob()`
in app.js), terminated by the next request, inline where there is no
`Worker`. A failed search is said on the panel and rethrown.

## One variable at a time
Collapsed beneath the results: each variable alone in every round, scored
on every played season before the displayed one, with a chance row. No
single variable reproduces the Final Four in more than one season (2025,
the all-1-seed year); the best (Torvik national rank) gets 26 of 56 Final
Four teams and 5 of 14 champions, seed 22 and 3, chance 3.5 and 0.2. The
rows are not controls.

## What it is not
Not validated, not a model, never scored: no P(1st), no EV, no track
record. Experimental at every surface (table row, headline tag, panel,
export header). A rule that reproduces the last two seasons is a
description of those seasons found after the fact; the panel shows how
quickly that gives out and how few survivors repeat. Ranking survivors to
avoid composite ratings is a legibility choice, not a claim that a
specific variable predicts better than a rating one — it does not change
which seasons a rule reproduces, its generalisation, or its short-history
warning, only which of the equally-simple, equally-fitted survivors is
shown first.

## Tests
`tests/test_collinearity.js`: rulePlay's tie rule; sequences against brute
force; `ruleRun` growing back, stopping, skipping, empty and throwing
cases; the cap pruned inside the enumeration equal to filtering after;
ranking under and without the cap; the specific-first tie-break (a rating
key at the position code order would otherwise favor still ranks below
every rating-free survivor, at any complexity, and never changes which
sequences survive); the 32-key sign-bit edge under the cap;
`ruleGeneralisation`; the job (distinct by picks / by sequence, the run and
what lies outside it, a skipped newest season, the named rule in all four
outcomes, no seasons). `tests/test_picks_export.js`: URL round-trip and
junk, defaults, the checkpoint guard with its message, the run across
2026/2024/2023 displays, the skip case, the one-variable table, the worker
protocol (same job, termination, error), the redraw, Flexible resetting the
choice, the chosen rule across seasons, the pending field and the choice
carrying to the field, the overlapping call. `tests/e2e/test_site_smoke.py`
(Chromium over docs/): the pending landing on desktop and phone, a chosen
rule said missing on 2026, the collapsed one-variable table, filter gating,
the guard's message, the skipped-season state on 2025, the search off the
main thread.
