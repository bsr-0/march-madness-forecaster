# Bracket Lab UI Council — Summary (2026-09-22)

Full deliberation: `council-report-20260922.md` (5-advisor panel + peer review).

## Convergence (4/5 advisors)

The real problem isn't raw content volume — it's **sequencing**. The page asks a
first-time visitor to choose between jargon-laden strategies ("pool-optimal" vs
"EV-optimal" vs "fitted model") before they've ever seen a bracket.

Fix: show a bracket already generated with a sensible default up front, and
consolidate the scattered customization surfaces (strategy cards, 4 filter
sub-panels, rule search, explore-a-variable) into one or two unified entry
points instead of ~8 stacked accordions.

## The clash

- **Executor**: ship a reorder/merge this week — mostly DOM reshuffling given
  the existing `<details>`-based architecture.
- **Statistician / Contrarian**: several "simplification" passes have already
  shipped on intuition with no usage data behind them; instrument accordion
  open-rates and scroll-depth before guessing again.

Peer review sided with this critique unanimously — all 3 reviewers flagged the
Executor's plan as reshuffling the deck without questioning whether
strategy-picker-before-bracket is even the right order.

## Recommendation

Do both, in sequence: ship the Executor's reorder (cheap, reversible), but add
open-rate instrumentation (a few `onclick` handlers on the `<details>`
elements) in the same pass, so next season's decisions aren't another guess.

## Blind spots flagged by peer review (≥2/3 reviewers)

- Nested `<details>`-in-`<details>` accessibility (screen readers, keyboard
  nav) was never addressed by any advisor.
- "Mobile" was treated as "smaller desktop," not a distinct interaction mode —
  no advisor discussed real touch-target sizing or mobile-specific testing.
