# Rule search — experimental strategy (2026-09-17)

Not a methodology change. A fenced, client-side feature on the site; nothing
in production scoring, selection, the candidate artifact, the fitted
evaluation or the track record reads it.

## What it does
For the displayed season Y, the user chooses checkpoints (any of Elite
Eight, Final Four, finalists, champion — at least one of the first two, see
below) and a fit range: any `from`–`to` span of played seasons strictly
before Y, with "last 3" / "last 4" shortcuts (default: last 3). The
eligible criteria default to every variable plus seed and can be narrowed
by checkbox. `docs/fit.js ruleSearch()` enumerates every
one-criterion-per-round rule — in each round every game goes to the team
better on one variable (direction-corrected z; ties to the better seed, then
lower index, the board's own rule) — and keeps those reproducing the
checkpoints in every fit season. If none survive it backs off one season at
a time and reports the longest range that does. Surviving rules are ranked
simplest first (distinct criteria, then switches, then criterion order round
by round). The user picks how many distinct brackets for season Y to offer
(1–20, default 5) and how to rank them: simplest first, or by the number of
played seasons *outside* the seasons the rule was selected on that it also
reproduces (ranked among the 500 simplest distinct brackets). That outside
count is shown on every offered bracket and is the one figure not selected
on; after a back-off the dropped seasons count as outside, and as misses.

**Compose by hand.** A second mode with no search: the user picks one
criterion per round (starting from the season's first listed variable in
every round), the page applies it to season Y and lists every played season
before Y, marking those in which the rule reproduces the chosen checkpoints.
Nothing is selected on, so the per-season marks are plain description.

**Board annotation.** While this strategy is showing, the criterion that
decided each round is printed under the round's header (and in the mobile
round navigator).

**Checkpoint guard.** At least one of Elite Eight / Final Four must stay
checked. Without an early checkpoint the search cannot prune before the last
constrained round and would enumerate criteria^rounds sequences per season
(32^5 with only the champion), which is not feasible in a browser. The panel
states the constraint; unchecking the last of the two is refused.

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

## What it is not
Not validated, not a model, never scored: no P(1st), no EV, no track record.
The page labels it experimental at every surface (table row, headline tag,
panel, export header). Measured on the shipped data the regress is stark and
is shown, not hidden: 2026 alone needs 2 criteria, 2025–2026 needs 3, and
2023–2025 or 2024–2026 have no surviving rule at all.

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
fetch stub: range clamping and defaults, the checkpoint guard, eligible-key
handling, search with back-off and outside-range counting, compose-by-hand
with its per-season marks, and the overlapping-call token).
