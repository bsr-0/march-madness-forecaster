# Rule search — experimental strategy (2026-09-17)

Not a methodology change. A fenced, client-side feature on the site; nothing
in production scoring, selection, the candidate artifact, the fitted
evaluation or the track record reads it.

## What it does
For the displayed season Y, the user chooses checkpoints (any of Elite
Eight, Final Four, finalists, champion) and a range (the last 3 or 4 played
seasons strictly before Y). `docs/fit.js ruleSearch()` enumerates every
one-criterion-per-round rule — in each round every game goes to the team
better on one variable (direction-corrected z; ties to the better seed, then
lower index, the board's own rule) — and keeps those reproducing the
checkpoints in every fit season. If none survive it backs off one season at
a time and reports the longest range that does. Surviving rules are ranked
simplest first (distinct criteria, then switches); the first five distinct
brackets they give season Y are offered. Each shows the number of played
seasons *before* the fit range it also reproduces — the one figure not
selected on.

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
intersection and back-off, ranking, tie rule, bracket application);
`tests/test_picks_export.js` (URL round-trip, no p1/ev/record on the row or
strategy, resolves to nothing without a result, wording guard extended to the
panel).
