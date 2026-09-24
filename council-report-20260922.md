# Council Report — 2026-09-22 Bracket Lab UI simplification

## Question

The user runs "Bracket Lab," a March Madness bracket forecasting site (`docs/index.html`/`app.js`/`app.css`). They asked: how do I improve the UI, keeping all existing features but displaying them more intuitively, on both desktop and mobile? Is the current UI too jam-packed with options? How can I simplify the layout while keeping the customization options for users?

## Metadata

- Bucket: **E** (genuinely novel — no prior `COUNCIL_LESSONS.md` or council transcripts found in this repo)
- Panel: 5 advisors (Contrarian, First Principles Thinker, Statistician, Outsider, Executor)
- Peer review: **ran (3 reviewers)** — Step 2.5 found material disagreement (ship-now vs. measure-first) between advisors
- Timestamp: 2026-09-22

## Per-Advisor Key Point + Action

- **The Contrarian** — Key point: prior "simplification" passes (accordions, conditional visibility) hid complexity rather than removing it, and the actual differentiator (the "Adjust this bracket" filters) is now the least discoverable thing on the page. Action: instrument the `<details>` sections with open-rate analytics before making any further layout change.
- **The First Principles Thinker** — Key point: the page is organized around "here are all the things this tool can do" instead of "here is your bracket, and here is optionally why." Action: show a bracket already generated with a sensible default up front, and consolidate strategy choice, filters, and rule search into one "Customize" surface with tabs instead of five stacked sections.
- **The Statistician** — Key point: "is the UI too jammed" conflates two unproven claims — that the page has a lot of content, and that the content confuses visitors — and no evidence yet supports the second. Action: instrument `<details>` open rates, scroll depth, and time-to-first-bracket-view split by device and by new-vs-returning visitor before redesigning.
- **The Outsider** — Key point: a first-time visitor is asked to choose between three jargon-laden strategy labels ("pool-optimal," "EV-optimal," "fitted model") before ever seeing a bracket, which requires domain fluency they don't have. Action: show a bracket first with a sensible default, and rename the strategies around outcomes ("best chance to win my pool") rather than methodology terms.
- **The Executor** — Key point: this is a reorder/merge job, not a rebuild — the `<details>`-based architecture already supports it. Action: this week, put strategy picker + board + a single "Adjust" entry point above the fold, merge the four filter sub-panels into one tabbed panel, merge Rule Search and Explore-a-Variable under one "Advanced" disclosure, and default more panels to closed on mobile.

## Where Advisors Agree Most

- **Sequencing is the core defect, not raw content volume.** Contrarian, First Principles, Outsider, and Executor all independently converge on: the page asks users to make an expert choice (which strategy) before showing them the payoff (a bracket). Fix the order, not just the density.
- **Consolidate scattered controls into fewer entry points.** First Principles, Outsider, and Executor all propose collapsing the current 4+ separate customization surfaces (strategy cards, 4 filter sub-panels, rule search, explore-a-variable) into one or two unified surfaces (a single "Customize" panel with tabs), rather than further accordion-nesting.
- **Existing `<details>`-based architecture doesn't need to be rebuilt.** No advisor proposed a framework rewrite or new state-management layer; all proposed reordering/merging what's already there.

## Where Advisors Clash

- **Ship now vs. measure first.** The Executor argues for shipping a reorder/merge this week and treating it as basic maintenance ("Reorder, merge, default-collapse... that's a Monday project, not a June one"). The Statistician and Contrarian argue the team has already made several unvalidated "simplification" passes on intuition alone and should instrument analytics (accordion open rates, scroll depth, time-to-first-bracket) before shipping another layout change, to avoid repeating the same guess-and-ship pattern.
- **Whether the methodology section is clutter or trust infrastructure.** First Principles argues the "Why this bracket" methodology section should move *up*, near the bracket, because it's trust infrastructure, not clutter. The Executor's plan keeps it below the fold along with "everything else," treating it the same as the more peripheral rule-search/explore-a-variable tools.

## Blind Spots from Peer Review

- **The Executor's plan doesn't question its own premise.** All 3 reviewers picked Response E's (the Executor's) "biggest blind spot" as: it optimizes for shipping speed but never asks whether strategy-picker-before-bracket is the right default flow in the first place — it would ship a tidier version of the same sequencing problem the other advisors diagnosed.
- **"Mobile" was treated as "smaller desktop," not a distinct interaction mode.** 2 of 3 reviewers flagged that no advisor addressed real touch-target sizing, keyboard/screen-reader navigation through nested `<details>`-in-`<details>`, or mobile-specific interaction testing — everyone's mobile fix was "the same components, more default-collapsed."

## Load-Bearing Assumptions

- **Scoring encoding**: N/A (not a modeling question)
- **Opponent field**: N/A
- **RNG / sample count**: N/A
- **Year scope**: N/A
- **Baseline anchor**: N/A
- **Data sources**: Council was grounded in a direct read of `docs/index.html` (structure/HTML comments), `docs/app.css` (media-query breakpoints), and `docs/SITE_REVIEW_TODO.md` (prior simplification history and known open gaps) as of 2026-09-22; no analytics/usage data exists on this repo to ground the "too jammed" claim empirically — this is the Statistician's central point.

## Framed Question + Prior Art

The framed question sent to all five advisors described the current page structure top-to-bottom (header/season picker → strategy comparison cards → leaderboard → headline → bracket board → rule search panel (collapsed, experimental, one-strategy-only) → explore-a-variable panel (collapsed, fitted-model-only) → "Adjust this bracket" (collapsed, 4 chip sub-panels + near-tied-alternates) → methodology section (collapsed) → team-detail drawer), the multiple simplification passes already done (per/variable toggle removed, table→cards, `<details>` accordions, strategy-conditional filter visibility, dedicated mobile swipe board), and the known open gaps from the site's own review doc (no pool-settings customization, no lock-and-reoptimize, no multi-entry support). No prior council transcripts or `COUNCIL_LESSONS.md` existed in this repo, so there was no prior-art block to inject.

## Advisor Responses

### The Contrarian
Here's what nobody's saying out loud: you've already run the "declutter" playbook — cards, accordions, conditional visibility, dedicated mobile board — and the page is still one long vertical stack with a team-detail drawer and four sub-panels of chips inside a collapsed section inside a page. Simplification passes that only move things into `<details>` don't reduce complexity, they hide it. The complexity is still there; you've just made the user pay a tap tax to discover it exists. That's not simpler, it's a worse information scent.

Three specific failure modes I'd bet money on:

1. **The strategy picker is upstream of everything but isn't load-bearing enough.** A first-time visitor picks a strategy before they understand what a strategy even changes downstream (filters appear/disappear, rule-search appears/disappears). You're asking for a commitment before showing the payoff. That's backwards — most users will pick randomly or bail.
2. **"Adjust this bracket" with four chip sub-panels collapsed by default means your actual differentiator — customization — is the least discoverable thing on the page.** If customization is the value prop, burying it under a `<details>` fold is optimizing for the wrong metric (perceived simplicity) at the cost of the real one (usage of the thing you built).
3. **You have no data on drop-off.** You cite one measured null (the toggle). Where's funnel instrumentation on the accordions — how many people ever open "Adjust this bracket" at all? Without that, every other change here is a guess dressed up as a redesign. Ship analytics before you ship layout changes, or you'll be back here next season re-litigating this blind.

### The First Principles Thinker
The real question isn't "how do I lay this out better" — it's "what is this page for, and does every element earn its place on the critical path to that?" You're treating UI density as a layout problem. It's actually a mission-clarity problem wearing a layout costume.

Strip it down: a first-time visitor during tournament season has exactly one job to do — get a bracket they trust. Everything else — rule search, variable explorer, filter sub-panels, methodology equations — is not the product, it's provenance for the product. You've been organizing the page around "here are all the things this tool can do" when it should be organized around "here is your bracket, and here is optionally why."

That reframe changes the fix from "reduce clutter" to "change the default object in view." Right now the page's implicit answer to "what am I looking at first" is a strategy-selection quiz. It should be a bracket, already generated with a sensible default (pool-optimal), visible in under one scroll, with everything else — strategy switch, filters, rule search, methodology — demoted to a single "Customize" affordance that's structurally identical to your `<details>` pattern but consolidated into one entry point instead of eight parallel ones scattered down the page.

Don't ask "which accordions do I collapse further." Ask "why are there four separate places a user could plausibly go to change something." Merge strategy-choice, filters, and rule-search into one customization surface with tabs, not five stacked sections competing for the same scroll. The methodology section isn't clutter to hide — it's trust infrastructure and belongs near the bracket, not buried below tools. Simplify the map, not the rooms.

### The Statistician
The framing "is the UI too jammed" is unanswerable from the description alone — it's a page audit, not an evidence question. You've done several simplification passes justified by intuition (and one "measured null" toggle removal, which tells me you *can* measure, but haven't applied it here). Before touching layout again, separate two claims that are being conflated: (1) the page has a lot of content, and (2) that content confuses or repels first-time visitors. Only the second matters, and nothing in your write-up demonstrates it.

Get actual data before redesigning:
- Instrument the `<details>` elements (onclick/toggle events fire fine even though state lives in HTML) to see open rates per section on real visits. If Rule Search and Explore-a-Variable are opened by <2% of sessions, that's evidence for demoting them, not intuition.
- Scroll-depth and time-to-first-bracket-view, split by device. You suspect mobile is the pain point but haven't shown it — six-column desktop and swipeable mobile may have very different real bounce/engagement profiles.
- Segment new vs. returning visitors (cookie or localStorage flag). A "jammed" page is a different problem for a first-timer than for a returner who wants their filters.

Watch selection bias: don't let power users (who already have opinions in a feedback form) dominate your read of "confusing." Watch multiple-comparisons: if you A/B five layout variants against a small March-only traffic window, you will find a "winner" by noise alone — pre-register one hypothesis, not five.

If you ship a redesign without this, you're replacing one unvalidated layout with another unvalidated layout, dressed up as progress.

### The Outsider
I have no idea what a "pool-optimal vs EV-optimal strategy" means, what a "fitted statistical model" is, or why a bracket needs a "rule search" tool. I'm the person who lands on this page during a coffee break because a coworker mentioned March Madness. Here's what happens to me:

I hit a header and a countdown — fine, I know what that is. Then three "strategy" cards ask me to choose between options whose names mean nothing to me. I haven't seen a bracket yet. I'm already asked to make an expert decision before I've seen the product work. That's the real problem, not information density — it's sequencing. You're front-loading a choice that requires domain fluency I don't have.

My blunt fix: show me a bracket FIRST, already filled in with a sensible default, no choice required. Let me see the thing working. Then, and only then, surface "want to change how this was picked?" as a single, low-commitment entry point into strategy selection — not three cards competing for my attention on load.

Second thing that would genuinely confuse me: "pool-optimal," "EV-optimal," and "fitted model" all sound like the same thing described three different ways to someone outside your field. If I can't tell from the label alone why I'd pick one over another, the label has failed, no matter how good the methodology writeup is later. Rename them around outcomes I recognize ("best chance to win my pool," "closest to the smartest picks," "matches historical patterns") — or don't ask me at all until I've asked "why did it pick this."

You've simplified the box. I'm telling you the box comes first when it should come last.

### The Executor
Stop describing the problem more elegantly — pick a structure and ship it Monday. Here's the first step, literally:

1. **Above the fold gets three things only: strategy picker, board, "Adjust" entry point.** Everything else — leaderboard, methodology, rule search, explore-a-variable — moves below the fold or behind one nav tab. That's a CSS/markup reorder, not a rebuild. You already have `<details>` for everything, so this is mostly reordering DOM elements and adjusting default-open state. Half a day.
2. **Collapse the four filter sub-panels into one panel with a tab strip inside it**, not four stacked `<details>`. Users don't scan four accordions to find "biggest upset in Final Four" — they want one "Adjust" button that opens a single surface with categories inside. Still all HTML/CSS, no JS state needed if you use radio-button-driven CSS tabs.
3. **Rule search and Explore-a-variable get merged under one "Advanced" disclosure**, gated behind a single click, not two separate orange boxes competing for attention. If usage on those is near-zero (check your analytics now — do you have any?), that's your evidence to cut them further next season.
4. **Mobile: don't redesign, just change disclosure defaults.** Same HTML, same components — on mobile, more things default to closed, that's it. Use a media query on the `open` attribute via JS if you must, but even a CSS-only "hide filters panel below 720px until tapped" ships without touching app.js.

Do NOT start with "let's rethink the information architecture." That's a June project, not a Monday one. Reorder, merge, default-collapse. Ship that, then measure before you touch strategy or pool-settings features — those are new work, not UI cleanup, and they're a different ticket.

## Peer Review

**Reviewer 1**
1. Strongest: **B** (First Principles). It's the only response that names the actual mechanism (page has no clear "job to be done," so nothing has priority) and gives an actionable structural fix — default-generated bracket + single "Customize" surface — rather than just reordering existing chrome (E) or demanding more data before deciding anything (C).
2. Biggest blind spot: **E** (Executor). It optimizes for shipping fast by reordering/merging what exists, but never questions whether "strategy picker first" is even the right default experience — it inherits A/B/D's diagnosed sequencing problem without fixing it.
3. All missed: none discuss the team-detail drawer's interaction with the rest of the page on mobile, none address information hierarchy *within* the bracket board itself (treated as a monolith), and none address accessibility of nested `<details>`-in-`<details>` for screen readers/keyboard nav.

**Reviewer 2**
1. Strongest: **D** (Outsider). It's the only response that inhabits an actual user's head and produces a concrete, checkable failure (jargon-laden strategy labels before any payoff shown), with both a sequencing fix and a naming fix.
2. Biggest blind spot: **E** (Executor). It optimizes purely for shipping speed and treats the redesign as mechanical reordering/merging, but never questions whether the strategy-picker-first sequencing (which A and D both flag as the core problem) is even right.
3. All missed: none sketched what the simplified page actually looks like at narrow widths despite the question explicitly asking about desktop vs. mobile parity; none propose a genuinely different interaction pattern beyond "collapse/merge/reorder" or "show data first," despite the team's stated context that shallow fixes (accordions, conditional visibility) have already been tried.

**Reviewer 3**
1. Strongest: **D** (Outsider). Grounded in a concrete persona, identifies a real, testable UX flaw (jargon indistinguishable to a new user) with actionable rename suggestions, where B makes the same reframe more abstractly.
2. Biggest blind spot: **E** (Executor). Optimizes for shipping speed but never questions whether the underlying IA (strategy picker before bracket, four separate customization sites) is the actual problem — sidesteps the most substantive critique raised by its own peers.
3. All missed: no discussion of accessibility/keyboard navigation or real mobile touch-target testing (mobile treated as "smaller desktop"); no discussion of what happens to returning/power users who rely on the current dense layout; no competitive benchmarking against how other bracket tools solve this exact sequencing problem.

## Footer

Counciled 2026-09-22: Bracket Lab UI simplification (march-madness-forecaster repo). 5-advisor panel, peer-reviewed (diverged on ship-now vs. measure-first).
