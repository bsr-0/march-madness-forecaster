---
name: llm-council
description: "Run any question, idea, or decision through a council of 5 AI advisors who independently analyze it, peer-review each other anonymously, and synthesize a final verdict. Based on Karpathy's LLM Council methodology. MANDATORY TRIGGERS: 'council this', 'run the council', 'war room this', 'pressure-test this', 'stress-test this', 'debate this'. STRONG TRIGGERS (use when combined with a real decision or tradeoff): 'should I X or Y', 'which option', 'what would you do', 'is this the right move', 'validate this', 'get multiple perspectives', 'I can't decide', 'I'm torn between'. Do NOT trigger on simple yes/no questions, factual lookups, or casual 'should I' without a meaningful tradeoff (e.g. 'should I use markdown' is not a council question). DO trigger when the user presents a genuine decision with stakes, multiple options, and context that suggests they want it pressure-tested from multiple angles."
---

# LLM Council

You ask one AI a question, you get one answer. That answer might be great. It might be mid. You have no way to tell because you only saw one perspective.

The council fixes this. It runs your question through 5 independent advisors, each thinking from a fundamentally different angle. Then they review each other's work. Then a chairman synthesizes everything into a final recommendation that tells you where the advisors agree, where they clash, and what you should actually do.

This is adapted from Andrej Karpathy's LLM Council. He dispatches queries to multiple models, has them peer-review each other anonymously, then a chairman produces the final answer. We do the same thing inside Claude using sub-agents with different thinking lenses instead of different models.

---

## When to Run the Council

The council is for questions where being wrong is expensive.

Good council questions:
- "Should I launch a $97 workshop or a $497 course?"
- "Which of these 3 positioning angles is strongest?"
- "I'm thinking of pivoting from X to Y. Am I crazy?"
- "Here's my landing page copy. What's weak?"
- "Should I hire a VA or build an automation first?"

Bad council questions:
- "What's the capital of France?" (one right answer, no need for perspectives)
- "Write me a tweet" (creation task, not a decision)
- "Summarize this article" (processing task, not judgment)

The council shines when there's genuine uncertainty and the cost of a bad call is high. If you already know the answer and just want validation, the council will likely tell you things you don't want to hear. That's the point.

---

## The Five Advisors

Each advisor thinks from a different angle. They're not job titles or personas. They're thinking styles that naturally create tension with each other.

### 1. The Contrarian
Actively looks for what's wrong, what's missing, what will fail. Assumes the idea has a fatal flaw and tries to find it. If everything looks solid, digs deeper. The Contrarian is not a pessimist. They're the friend who saves you from a bad deal by asking the questions you're avoiding.

### 2. The First Principles Thinker
Ignores the surface-level question and asks "what are we actually trying to solve here?" Strips away assumptions. Rebuilds the problem from the ground up. Sometimes the most valuable council output is the First Principles Thinker saying "you're asking the wrong question entirely."

### 3. The Statistician
Examines the methodology, not the conclusion. Checks sample sizes, multiple comparisons, distributional assumptions, selection bias, temporal stationarity, and whether the statistical machinery actually supports the claims being made. The Statistician doesn't care about the business case or the strategy. They care about whether the numbers mean what you think they mean. If the experiment is flawed, nothing downstream matters.

### 4. The Outsider
Has zero context about you, your field, or your history. Responds purely to what's in front of them. This is the most underrated advisor. Experts develop blind spots. The Outsider catches the curse of knowledge: things that are obvious to you but confusing to everyone else.

### 5. The Executor
Only cares about one thing: can this actually be done, and what's the fastest path to doing it? Ignores theory, strategy, and big-picture thinking. The Executor looks at every idea through the lens of "OK but what do you do Monday morning?" If an idea sounds brilliant but has no clear first step, the Executor will say so.

**Why these five:** They create three natural tensions. Contrarian vs Executor (what will fail vs just do it). First Principles vs Outsider (strip to fundamentals vs fresh eyes on the surface). The Statistician keeps everyone honest by checking whether the evidence actually supports the claims — the advisor most likely to catch the blind spots that peer review previously had to surface.

---

## How a Council Session Works

### Step 0: Pre-Council Duplicate Check (MANDATORY — do NOT skip)

**The council is expensive. Eleven sub-agents per session. Prior sessions
on this repo repeated the same question 6× in different wording ("biggest
limitation?", "most critical gap?", "single most critical limitation?"
were all the same question). The user lost agent-time and cycles
re-debating settled ground.**

Before doing ANYTHING else, check whether this question has already been
answered or is already identified as an open item.

**A. If the workspace has a `COUNCIL_LESSONS.md` file** (or equivalent
consolidated-council-lessons doc), the protocol is defined in that file's
`§4 Pre-Council Duplicate-Check Protocol`. Read that section and follow
it literally. It classifies the user's question into one of five buckets
(A = already answered, B = already-identified open item, C = blocked on
prerequisite, D = locked decision, E = genuinely novel) and tells you
whether to abort or continue.

**B. If the workspace has no `COUNCIL_LESSONS.md`** but there are
`council-transcript-*.md` files present, scan their headings/questions
for obvious duplicates (same wording, same semantic shape) before
proceeding. If you find 2+ transcripts asking the same question, flag to
the user: "You've counciled this on `<dates>`. Summarise prior verdicts
before we re-run?"

**C. If neither is present**, proceed directly to Step 1 (no duplicate
state to check).

**When you abort in Step 0, do so politely and specifically:**
- Name the matched §3 row or §2 `Ox` ID.
- Quote the one-line verdict or gate.
- Ask the user: "Do you have new evidence that invalidates this, or should
  we move to executing the existing verdict?" Then stop and wait.

**Do NOT summon the council when the real blocker is execution of an
already-identified item.** Re-debating O1 ("collect 31 real brackets")
will not collect the brackets.

### Step 1: Frame the Question (with context enrichment)

When you reach this step (bucket E, or bucket A/D with explicit new
evidence), do two things before framing:

**A. Scan the workspace for context.** The user's question is often just the tip of the iceberg. Their Claude setup likely contains files that would dramatically improve the council's output. Before framing, quickly scan for and read any relevant context files:

- `CLAUDE.md` or `claude.md` in the project root or workspace (business context, preferences, constraints)
- `COUNCIL_LESSONS.md` or `MEMORY.md` (consolidated past decisions, locks, open items — these bound the space of acceptable proposals)
- Any `memory/` folder (audience profiles, voice docs, business details, past decisions)
- Any files the user explicitly referenced or attached
- Any other context files that seem relevant to the specific question (e.g., if they're asking about pricing, look for revenue data, past launch results, audience research)

Use `Glob` and quick `Read` calls to find these. Don't spend more than 30 seconds on this. You're looking for the 2-3 files that would give advisors the context they need to give specific, grounded advice instead of generic takes.

**Prior-art injection (per §4 Step 3 of `COUNCIL_LESSONS.md`, when
present):** when framing the question for the 5 advisors, include a
"Prior art" block listing related `Ox` open items, locked lessons from
§1, and the 2-3 most related §3 rows by keyword match. This prevents
advisors from re-deriving settled ground and focuses their work on the
genuinely novel part of the question.

**B. Frame the question.** Take the user's raw question AND the enriched context and reframe it as a clear, neutral prompt that all five advisors will receive. The framed question should include:

1. The core decision or question
2. Key context from the user's message
3. Key context from workspace files (business stage, audience, constraints, past results, relevant numbers)
4. What's at stake (why this decision matters)

Don't add your own opinion. Don't steer it. But DO make sure each advisor has enough context to give a specific, grounded answer rather than generic advice.

If the question is too vague ("council this: my business"), ask one clarifying question. Just one. Then proceed.

Save the framed question for the transcript.

### Step 1.5: Choose the Panel Size (bucket-driven)

The 5-advisor panel is the full treatment. It is not always the right tool.
The council is expensive — 11 sub-agents at full scale (5 drafts + 5
peer reviews + 1 chairman). Use the bucket classification from Step 0 /
`COUNCIL_LESSONS.md §4` to pick the right size:

| §4 bucket | Advisors | Panel composition | Why |
|---|---|---|---|
| **A** (already answered) + user has new evidence | 2 | Contrarian + Statistician | Re-litigating settled ground — need the skeptic and the numbers check, not a fresh 5-angle fan-out |
| **B** (execute an open `Ox`) | 1 | Executor | Strategic question is settled; just sanity-check the plan |
| **C** (blocked on prereq) | 0 | — | Step 0 should have aborted — you shouldn't be here |
| **D** (locked decision) + user has new evidence | 2 | Contrarian + Statistician | Same rationale as A |
| **E** (genuinely novel) | 5 | All five | What the council was designed for |

Rules of thumb:

- **Default to full panel (5) when in doubt.** The cost of false-negative
  (under-counciling a real novel question) is much higher than the cost
  of false-positive (over-counciling a bucket-A question).
- **Never go below the size prescribed by the bucket.** If the user
  insists on a "quick council" of the full bucket-E panel, that's bucket
  E — run all 5.
- **Record the panel size in the transcript** ("Panel: 2 advisors
  (Contrarian + Statistician); bucket A with new evidence"). This lets
  future Step 0 duplicate-check see what kind of verdict was produced.

### Step 2: Convene the Council (run the panel in parallel)

Spawn the advisors chosen in Step 1.5 **in a single tool-call batch** —
one message with N `Agent` tool uses, dispatched together. Do NOT run
them sequentially. Sequential spawning is where most wall-clock cost
lives, and the advisors are independent by design — they have no data
dependency on each other. A 5-way parallel fan-out completes in the
time of the slowest advisor, not the sum.

Each advisor gets:

1. Their advisor identity and thinking style (from the descriptions above)
2. The framed question + prior-art block from Step 1
3. A clear instruction: respond independently. Do not hedge. Do not try
   to be balanced. Lean fully into your assigned perspective. If you see
   a fatal flaw, say it. If you see massive upside, say it. Your job is
   to represent your angle as strongly as possible. The synthesis comes
   later.

Each advisor should produce a response of **150-250 words**. Tight
enough to force a top-3 thesis rather than an enumeration. If an
advisor can't make their case in 250 words, they're hedging — the
shorter cap surfaces that.

**Sub-agent prompt template:**
```
You are [Advisor Name] on an LLM Council.

Your thinking style: [advisor description from above]

A user has brought this question to the council:

---
[framed question]
---

Respond from your perspective. Be direct and specific. Don't hedge or try to be balanced. Lean fully into your assigned angle. The other advisors will cover the angles you're not covering.

Keep your response between 150-250 words. No preamble. Go straight into your analysis.
```

### Step 2.5: Check Convergence (skip peer review if advisors agree)

Read all advisor responses. Extract each advisor's bottom-line
recommendation — usually the last paragraph or an explicit "so do X"
line. Compare them.

**Converge test:** do the advisors land on the same recommendation
(same action, same direction, same scope)?

- **5-advisor panel (bucket E):** if ≥ 4 of 5 advisors converge on the
  same recommendation AND no advisor calls the majority view a "fatal
  flaw", the council agrees. Peer review will add nothing except cost
  and marginal blind-spot detection. **Skip to Step 4** with a
  `peer_review: skipped (convergence)` note in the transcript.
- **2-advisor panel (bucket A / D):** if both advisors agree, go
  straight to Step 4. If they disagree, skip peer review anyway —
  there's no "majority" to peer-review with only 2 voices. The
  chairman directly mediates the disagreement.
- **1-advisor panel (bucket B):** peer review is always skipped. Go
  straight to Step 4.

**Diverge case (only on 5-advisor panel):** if 2+ advisors materially
disagree with the rest, or any advisor flags a "fatal flaw" in the
majority view, **proceed to Step 3**. Peer review is what Karpathy's
method exists for; this is the case it was designed to handle.

Record the convergence decision in the transcript so later Step 0
duplicate-checks can see whether the verdict came from a peer-reviewed
deliberation or a short-circuited unanimous first pass.

### Step 3: Peer Review (5 sub-agents in parallel, ONLY on divergence)

**Run this step only if Step 2.5 determined that the 5-advisor panel
has material disagreement.** Otherwise skip to Step 4.

This is the step that makes the council more than just "ask 5 times." It's the core of Karpathy's insight.

Collect all 5 advisor responses. Anonymize them as Response A through E (randomize which advisor maps to which letter so there's no positional bias).

Spawn 5 new sub-agents, one for each advisor. Each reviewer sees all 5 anonymized responses and answers three questions:

1. Which response is the strongest and why? (pick one)
2. Which response has the biggest blind spot and what is it?
3. What did ALL responses miss that the council should consider?

**Reviewer prompt template:**
```
You are reviewing the outputs of an LLM Council. Five advisors independently answered this question:

---
[framed question]
---

Here are their anonymized responses:

**Response A:**
[response]

**Response B:**
[response]

**Response C:**
[response]

**Response D:**
[response]

**Response E:**
[response]

Answer these three questions. Be specific. Reference responses by letter.

1. Which response is the strongest? Why?
2. Which response has the biggest blind spot? What is it missing?
3. What did ALL five responses miss that the council should consider?

Keep your review under 200 words. Be direct.
```

### Step 4: Chairman Synthesis

This is the final step. One agent gets everything: the original
question, all advisor responses (de-anonymized so you can see which
advisor said what), and — when Step 3 ran — all peer reviews.

**Inputs by panel size:**

- **5-advisor panel, converged (Step 2.5 skipped peer review):** the
  chairman receives the 5 advisor drafts only. They must note in their
  verdict that peer review was skipped due to convergence, so the
  reader can calibrate confidence accordingly.
- **5-advisor panel, diverged (Step 3 ran):** the chairman receives
  advisor drafts + all 5 peer reviews. This is the classic Karpathy
  flow.
- **2-advisor panel (bucket A / D):** the chairman receives 2 advisor
  drafts. They must either endorse one side or carve out a specific
  middle that neither advisor argued. "Both have points" is not
  acceptable.
- **1-advisor panel (bucket B):** the chairman receives 1 advisor draft
  and their own independent read of the plan. Their job is to
  stress-test the Executor's sanity check, not to expand scope.

The chairman's job is to produce a verdict that no individual advisor
could have reached alone. The chairman is not a summarizer — the
transcript already exists for that. The chairman is the most senior
mind in the room, expected to make judgment calls, take sides on
disagreements, and add independent analysis.

**COUNCIL VERDICT**

1. **Where the council agrees** -- the points that multiple advisors converged on independently. These are high-confidence signals.

2. **Where the council clashes** -- the genuine disagreements. The chairman MUST pick a side and defend the choice. "It depends" and "the truth is somewhere in between" are not acceptable. If the minority has stronger reasoning, side with them.

3. **Blind spots the council caught** -- things that only emerged through the peer review round. Things individual advisors missed that other advisors flagged.

4. **The chairman's take** -- independent analysis. What did the council get wrong collectively? What would the chairman add that no advisor or reviewer raised? If the chairman agrees with consensus, they must explain what specific evidence makes that consensus trustworthy rather than just restating it.

5. **Critical actions** -- a maximum of 3 concrete actions, dependency-ordered (what must happen first gates what comes next). Each action is one sentence stating what to do, and one sentence stating the gate (how you know it's done or what result lets you proceed). No prose. No rationale. The reasoning lives in the sections above — this section is the punch list.

**Chairman prompt template (full 5-advisor, peer-reviewed):**
```
You are the Chairman of an LLM Council. Your job is to synthesize the work of 5 advisors and their peer reviews into a final verdict.

The question brought to the council:
---
[framed question]
---

ADVISOR RESPONSES:

**The Contrarian:**
[response]

**The First Principles Thinker:**
[response]

**The Statistician:**
[response]

**The Outsider:**
[response]

**The Executor:**
[response]

PEER REVIEWS:
[all 5 peer reviews]

Produce the council verdict using this exact structure:

## Where the Council Agrees
[Points multiple advisors converged on independently. These are high-confidence signals.]

## Where the Council Clashes
[Genuine disagreements. Do NOT just present both sides — make the call. State which side you believe is correct and why. If the minority position has stronger reasoning than the majority, side with it explicitly. "The truth is somewhere in between" is not acceptable — pick a side and defend it.]

## Blind Spots the Council Caught
[Things that only emerged through peer review. Things individual advisors missed that others flagged.]

## The Chairman's Take
[Your independent assessment. What did the council get wrong collectively? What would you add that no advisor or reviewer raised? If every advisor missed something, say it here. If you agree with the consensus, state what specific evidence convinced you — not just that "multiple advisors converged." You are not a summarizer. You are the sixth mind in the room.]

## Critical Actions
[Maximum 3 actions. Dependency-ordered — action 1 gates action 2, action 2 gates action 3. Each action is two sentences: what to do, and the gate (the concrete result that means it's done or the condition that unlocks the next action). No prose, no rationale — the reasoning lives in the sections above. This is the punch list the user takes away and executes. If only one action matters, list one. Do not pad to three.]

You are not a secretary taking minutes. You are the most senior person in the room. Summarizing what others said is not your job — that's what the transcript is for. Your job is to think independently, make judgment calls the advisors couldn't make individually, and deliver a verdict the user can act on without reading anything else. If you find yourself writing "the council agrees" without adding your own analysis of WHY that agreement is trustworthy (or suspicious), you're not doing your job.
```

**Chairman prompt template (5-advisor, converged — peer review skipped):**
```
You are the Chairman of an LLM Council. The council produced 5
independent advisor drafts; all converged on substantively the same
recommendation, so the peer-review step was skipped. Your job is to
produce the final verdict AND an independent red-team on why the
convergence should or should not be trusted.

Follow the full verdict structure above, with these adjustments:

- "Where the council agrees" is the entire panel — call out that the
  convergence is the signal.
- "Where the council clashes" becomes "Latent disagreement": any
  second-order tension in the drafts the advisors didn't surface as
  top-line. If there is none, write "No latent disagreement" — do NOT
  invent clashes.
- "Blind spots the council caught" becomes "Blind spots the chairman
  flags": what did all 5 miss that a peer-review pass likely would
  have caught? Be specific.
- "The chairman's take" must explicitly answer: is the convergence
  the signal, or is it groupthink? If groupthink, recommend re-running
  with peer review.
- "Critical actions" unchanged: max 3, dependency-ordered.
```

**Chairman prompt template (2-advisor, bucket A / D):**
```
You are the Chairman. Two advisors (Contrarian + Statistician) reviewed
a case where the user brought new evidence against a previously-settled
question. The question is:
---
[framed question]
---

Contrarian:
[response]

Statistician:
[response]

Produce a tight 3-section verdict (≤ 300 words total):

1. **What's the new evidence worth?** Side with one advisor or carve a
   specific middle neither argued. "Both have points" is NOT acceptable.
2. **Does the prior lock hold?** Yes / No / Needs-further-test. If
   "needs further test", name the specific test and the pass threshold.
3. **Critical actions** (max 2): what to do now; what to do next.
```

**Chairman prompt template (1-advisor, bucket B):**
```
You are the Chairman. The Executor reviewed a plan for executing an
already-identified open item:
---
[framed question]
---

Executor's sanity check:
[response]

Produce a tight verdict (≤ 200 words):

1. **Is the plan executable as stated?** Yes / No.
2. **Hidden prerequisites the Executor missed?** Name them or say
   "none spotted".
3. **Critical actions** (max 2): do X; then do Y.

Do NOT expand scope. Your job is to stress-test the sanity check, not
to re-open the strategic decision that §2 already captured.
```

### Step 5: Write the Council Report (single markdown file)

After the chairman synthesis is complete, write **one** markdown file that
serves as both the scannable report AND the full transcript. Do not
generate HTML, PDF, or any other format. Do not write a second
`council-transcript-*.md` file — this single file is the artifact.

**File:** `council-report-[timestamp].md`

**Why one markdown file, not an HTML report + markdown transcript:** the
old flow streamed a self-contained HTML document with inline CSS,
collapsible `<details>` scaffolding, and visual grids, then wrote the
same advisor drafts/peer reviews/verdict a second time as a
`council-transcript-*.md`. Every byte of that HTML boilerplate and every
duplicated sentence is emitted token-by-token, which is exactly where
the long delay between the last chairman statement and the finalized
report was coming from. Markdown renders natively in the terminal and in
every editor the user opens it in, needs no CSS, and only has to be
serialized once.

**Required structure (top-to-bottom, in this order):**

1. `# Council Report — YYYY-MM-DD [topic slug]` (H1)
2. **Question** — the user's raw question, one short paragraph.
3. **Metadata block** — bucket (A/B/C/D/E), panel size + composition,
   peer review status (`ran` / `skipped (convergence)` / `skipped (panel
   size)`), timestamp.
4. **Critical Actions** — rendered as a blockquote callout (`>` prefix)
   near the top. Extract verbatim from the chairman's synthesis. This
   appears exactly once — do NOT repeat it inside the chairman's
   analysis below.
5. **Chairman's Verdict** — the four analysis sections (Where the
   Council Agrees, Where the Council Clashes, Blind Spots the Council
   Caught, The Chairman's Take) as H2s. Omit the Critical Actions
   subsection here since it is already rendered in step 4.
6. **Agreement / Disagreement Summary** — a small markdown table with
   one row per advisor: name, one-line bottom-line recommendation,
   aligned/dissenting marker. Keep it to 5 short rows (or N for smaller
   panels).
7. **Framed Question + Prior Art** — the enriched prompt that was sent
   to every advisor, plus the `Ox`/lessons block that was injected.
8. **Advisor Responses** — each advisor's full 150-250 word draft under
   an H3 (`### The Contrarian`, etc.). Use `<details><summary>` only if
   you want them collapsed by default; otherwise plain H3s are fine.
9. **Peer Review** — either all reviewer outputs under an H2, or a
   single line `_Peer review skipped (convergence)._` / `_Peer review
   skipped (panel size = N)._` with the justification.
10. **Footer** — one-line timestamp + what was counciled.

**Deduplication rule (unchanged, now enforced in one file):** every
piece of information appears exactly once. Critical Actions live only
in the callout (item 4). Advisor drafts live only in section 8. The
chairman's verdict never re-states advisor drafts verbatim.

**What NOT to do:**
- Do not emit any `<style>`, inline `style="..."`, or CSS.
- Do not wrap the document in `<html>` / `<body>`.
- Do not write a second file. `council-transcript-*.md` is deprecated —
  this single `council-report-*.md` replaces both.
- Do not open the file in a browser; the user reads markdown inline.

### Step 6: Update the Lessons Index

The single markdown file from Step 5 already contains the full
transcript content (framed question, bucket, panel size, advisor
drafts, peer review, chairman). The only remaining housekeeping is
updating the consolidated lessons index.

Recording the panel size and peer-review decision in the report's
metadata block lets future Step 0 dedup-checks tell whether a prior
verdict came from a full 5-advisor peer-reviewed deliberation, a
converged-no-review short-circuit, or a 2-advisor re-litigation. That
distinction matters when the user says "the council said X" — a
2-advisor bucket-A verdict has different evidentiary weight than an
11-sub-agent bucket-E verdict.

**A. Append to `COUNCIL_LESSONS.md` if present.** Per that file's §4 Step
4 update rule:

- Append one row to §3 with: next sequential `#`, date, one-line framed
  question, one-line verdict (extracted from the chairman's Critical
  Actions).
- If the verdict closes an open `Ox` item in §2 → move the item to §1 and
  leave `[closed <date>]` in §2.
- If the verdict opens a new unresolved item → add a new `Ox` row with
  the next free ID (never reuse, never renumber).
- If the verdict supersedes a §1 lesson → append a superseding bullet and
  mark the old one `[SUPERSEDED <date>]`. Do not rewrite.

This keeps the duplicate-check protocol (Step 0) effective for the next
session. A `COUNCIL_LESSONS.md` that isn't updated after each council
quickly loses its value as a dedup index.

**B. Periodic report consolidation (user-initiated).** Raw reports
accumulate fast. When the user asks to prune / consolidate / clean up
council artifacts, the consolidation pattern is: extract lessons + open
questions into `COUNCIL_LESSONS.md`, then delete the raw
`council-report-*.md` files (plus any legacy
`council-transcript-*.md` or `council-report-*.html` left over from the
old two-file flow). This is user-initiated, not automatic.

---

## Output Format

Every council session produces exactly one file:

```
council-report-[timestamp].md   # single markdown artifact: report + transcript
```

The top of the file is the scannable verdict (question, critical
actions, chairman's analysis, agreement table). The bottom holds the
framed question, advisor drafts, and peer review for later reference.
No HTML, no PDF, no second transcript file — the old
`council-report-*.html` + `council-transcript-*.md` split is
deprecated.

---

## Important Notes

- **Always spawn all 5 advisors in parallel.** Sequential spawning wastes time and lets earlier responses bleed into later ones.
- **Always anonymize for peer review.** If reviewers know which advisor said what, they'll defer to certain thinking styles instead of evaluating on merit.
- **The chairman MUST make judgment calls, not just summarize.** If 4 out of 5 advisors say "do it" but the reasoning of the 1 dissenter is strongest, the chairman should side with the dissenter and explain why. The chairman adds a "Chairman's Take" section with independent analysis — something no advisor raised. If the chairman finds themselves just reorganizing what others said, the synthesis has failed.
- **Don't council trivial questions.** If the user asks something with one right answer, just answer it. The council is for genuine uncertainty where multiple perspectives add value.
- **Keep the report in markdown.** The old HTML report was the slowest step in the pipeline — inline CSS, collapsible `<details>`, visual grids, and then a duplicated markdown transcript all had to be streamed token-by-token. A single markdown file cuts the finalization time roughly in half and renders cleanly everywhere the user reads it.
