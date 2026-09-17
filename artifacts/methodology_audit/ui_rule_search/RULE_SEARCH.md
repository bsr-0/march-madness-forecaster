# Rule search — experimental strategy (2026-09-17)

Not a methodology change. A fenced, client-side feature on the site; nothing
in production scoring, selection, the candidate artifact, the fitted
evaluation or the track record reads it.

## What it does
For the displayed season Y, the user chooses checkpoints (any of Round of
32, Sweet 16, Elite Eight, Final Four, finalists, champion — at least one of
the first four, see below) and a fit range: any `from`–`to` span of played seasons strictly
before Y, with "last 2" / "last 3" shortcuts (default: last 2 — 2024–2025
while 2026 is displayed, 2025–2026 once a 2027 bracket exists; never the
displayed season). Default checkpoints: Final Four, finalists, champion.
The eligible criteria default to every variable plus seed and can be
narrowed by checkbox.

**Panel (2026-09-17).** Three one-line controls — Rounds (chips), Seasons
(from–to), Criteria (count, opens the checklist) — with mode, how many to
offer, ranking and the criteria cap under a collapsed "More". `docs/fit.js ruleSearch()` enumerates every
one-criterion-per-round rule — in each round every game goes to the team
better on one variable (direction-corrected z; ties to the better seed, then
lower index, the board's own rule) — and keeps those reproducing the
checkpoints in every fit season. If none survive it backs off one season at
a time and reports the longest range that does. Surviving rules are ranked
simplest first (distinct criteria, then switches, then criterion order round
by round). The user picks how many distinct brackets for season Y to offer
(1–20, default 5), how to rank them: simplest first, or by the number of
played seasons *outside* the seasons the rule was selected on that it also
reproduces (ranked among the 500 simplest distinct brackets), and a cap on
the distinct criteria a rule may use (any, or at most 1/2/3; survivors are
ordered by that count first, so the cap is a cut-off in the ranked list,
and the panel says when it left nothing). That outside count is shown on
every offered bracket and is the one figure not selected on; after a
back-off the dropped seasons count as outside, and as misses.

**Compose by hand.** A second mode with no search: the user picks one
criterion per round (starting from the season's first listed variable in
every round), the page applies it to season Y and lists every played season
before Y, marking those in which the rule reproduces the chosen checkpoints.
Nothing is selected on, so the per-season marks are plain description.

**Board annotation.** While this strategy is showing, the criterion that
decided each round is printed under the round's header (and in the mobile
round navigator).

**Checkpoint guard.** At least one of Round of 32 / Sweet 16 / Elite Eight
/ Final Four must stay checked. Without an early checkpoint the search
cannot prune before the finalists and would enumerate criteria^4 sequences
per season before its first constraint (32^5 with only the champion), which
is not feasible in a browser. The panel states the constraint; unchecking
the last early one is refused, and a link without one keeps the default.
Measured on the shipped data (2023–2025, 32 criteria): Sweet 16 + champion
2.1 s, Elite Eight + champion 1.1 s, Round of 32 alone 0 ms with no
survivor — all inside the default Final Four/finalists/champion case
(~5 s in Node on this machine, 2,064,027 survivors).

**Shareable.** Every input above, the mode, the composed rule and which
offered bracket is showing travel in the URL hash (`rm rc rf rt rn rk rr rx
ri rh`), read back by `readRuleHash()` with the same guards as the controls;
criteria a link names that the displayed season lacks are dropped (eligible
set) or replaced by the season's first variable (hand rule) rather than
reaching `rulePlay()`.

**Fixed while adding this (2026-09-17).** The held result was never
invalidated by a season change and nothing started a search on load, so
switching seasons in this strategy showed the previous season's picks —
team indices into a different bracket — on the new board, and a `#s=rule`
link opened on an empty board. `ruleStrategy()` now refuses a result whose
key (season and every input) is not the current one, `setYear()` reruns the
search, and the panel shows only a current result.

## Performance
The enumeration itself is ~0.3 s per season on the shipped data (2,064,027
survivors for 2025 with Final Four/finalists/champion). The first version
then sorted those with a comparator that decoded and re-counted each rule on
every comparison — 15 s — and recomputed a season's set for every back-off
step. Now one numeric key per rule (`ruleComplexityOfCode`) is sorted in a
typed array and per-season sets are memoised across the back-off: 2023–2025
for 2026 runs in ~1 s natively (ordering verified identical to the old
comparator on all 2,064,027 rules). Controls fire the search without
awaiting; a request token makes an overlapping earlier call abandon so the
result on screen always matches the controls.

**Off the main thread (2026-09-17, review item 10).** The whole search-mode
job is one pure function, `ruleSearchJob()` in fit.js (ruleSearch, the
offered entries, the generalisation of every survivor, the ranking, the
resolution of the rule a link names). The page runs it in a Web Worker
built from a blob that imports fit.js by the page's own stamped URL
(`runRuleSearchJob()` in app.js), so the tab keeps painting and scrolling
and "Searching…" is actually seen; a new request terminates the worker in
flight; where there is no `Worker` (the node harness, a browser without
workers) the same function runs inline. A search that fails is said on the
panel and rethrown, never shown as "nothing found". Hand mode and the
one-variable table stay on the main thread (milliseconds).

## What it is not
Not validated, not a model, never scored: no P(1st), no EV, no track record.
The page labels it experimental at every surface (table row, headline tag,
panel, export header). Measured on the shipped data the regress is stark and
is shown, not hidden: 2026 alone needs 2 criteria, 2025–2026 needs 3, and
2023–2025 or 2024–2026 have no surviving rule at all.

## One variable, every round (2026-09-17)
The panel's first content, before any control: each variable applied on
its own in every round, scored on every played season before the
displayed one — Final Four teams right, seasons where the Final Four is
exact (all four), champions right — sorted by Final Four teams right, six
rows shown and the rest behind "All 32 variables", with a chance row
(picks that ignore the teams: 1/16 per Final Four slot, 1/64 for the
champion). Walk-forward like the search; nothing is selected on it.
Measured on the shipped data (14 seasons): no single variable reproduces
the Final Four in more than one season, and that one is 2025, when all
four 1 seeds reached it; the best (Torvik national rank) gets 26 of 56
Final Four teams and 5 of 14 champions, seed itself 22 and 3, chance 3.5
and 0.2. The copy states that gaps of a few teams between rows are noise:
a season's four slots move together, so n is seasons, not teams. This is
the finding the search rests on — a rule has to switch variables between
rounds to reproduce even two seasons — and it is shown first so it does
not have to be discovered through the controls. Clicking a row composes
that variable by hand in every round. `oneVariableTable()` in app.js,
computed once per displayed season from the payloads the search loads.

## Generalisation of every survivor (2026-09-17, review item 2)
Beside the survivor count the panel states how many of *all* the
surviving rules reproduce the checkpoints in at least one played season
outside the fit range (`ruleGeneralisation()` in fit.js: exact while
rules × outside seasons ≤ 200,000 applications, otherwise on the simplest
prefix, which the copy says). On the shipped data for 2027 (fit
2025–2026, Final Four + finalists + champion): **of 7,137 survivors, none
reproduces those in any of the other 12 seasons** (13 reproduce the Final
Four alone in one other season). Under the "most other seasons first"
ranking the per-row outside count is what the list was sorted by (best of
500) and the footer says it is selected on; the count over every survivor
is the one number no control tunes. Refusing to uncheck the last early
checkpoint now shows a message instead of silently not toggling; the
round chips carry full names and the ranking caveat is visible text, not
a tooltip (review items 7, 8).

## A season whose field is not out (2026-09-17)
From the day the previous season is played until Selection Sunday, the
newest listed season (`status: not_started` with played seasons before it)
is the page's front door: the strategy table lists the pool strategies and
the fitted model as awaiting the field (not selectable), the rule search
runs over the seasons before it (2025–2026 for 2027 by default — 2026 is a
*prior* season from 2027's point of view, which is why it belongs in
2027's fit and never in its own), and the board is 63 blank games under
the chosen rule's criteria. Variable keys and labels come from the latest
played season. Results are distinct *rules* (there is no bracket to
deduplicate by) with the outside-range count; picks are null and
`ruleStrategy()` resolves to nothing. The chosen rule travels in the URL as
its criterion sequence (`rq`), not a list position, so the same link
resolves to the bracket that rule gives once the field exists (matched by
sequence, then by picks since the list is then deduplicated by bracket;
a surviving rule not among the offered `n` is listed as one more). A
season change keeps the chosen rule; a control change resets it. The year
row lists the newest season first and the played seasons after a "past
seasons" label: they are the history every backtest claim is checked
against. The orientation line is the first thing on the page under a
pending field, and the board is six round headers with the chosen rule's
criterion and the game count rather than 63 empty games (review item 6).
A rule matched by the bracket it gives becomes the offered entry itself,
so the round labels name the rule the link names (item 9).

## Walk-forward
Fit seasons are strictly before the displayed season. A 2026 bracket from a
rule fit on 2024–2026 would be a bracket fit on its own result and is not
constructible from the UI.

## Tests
`tests/test_collinearity.js` (search semantics against brute force,
intersection and back-off, ranking as `ruleSearch()` actually emits it,
`ruleComplexityOfCode` against `ruleComplexity`, tie rule, bracket
application); `tests/test_picks_export.js` (URL round-trip, no p1/ev/record
on the row or strategy, resolves to nothing without a result, wording guard
extended to the panel; and, driven end to end on synthetic seasons through a
fetch stub: range clamping and defaults, the checkpoint guard including the
new early rounds, eligible-key handling, search with back-off and
outside-range counting, the criteria cap, compose-by-hand with its
per-season marks, the chosen bracket surviving a link's first search and
resetting after, and the overlapping-call token; plus the rule
configuration's URL round-trip with junk and missing-key handling, and the
regression that a result for another season or other inputs is not shown;
the search job on an 8-team fixture and the worker protocol through a
stub); `tests/e2e/test_site_smoke.py` (the rendered page in Chromium: the
pending landing on desktop and phone, a chosen rule carrying to 2026, hand
mode's result line, filter gating per strategy, the checkpoint guard's
message, the search off the main thread with the latest controls winning;
job `site-smoke` in CI).
