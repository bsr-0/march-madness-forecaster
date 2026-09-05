---
name: llm-council
description: "Run any question, idea, or decision through a council of 5 AI advisors who independently analyze it, peer-review each other anonymously, and surface a lightweight summary of each advisor's key point and action, where they agree, and where they clash — without a chairman agent picking sides. Adapted from Karpathy's LLM Council methodology. MANDATORY TRIGGERS: 'council this', 'run the council', 'war room this', 'pressure-test this', 'stress-test this', 'debate this'. STRONG TRIGGERS (use when combined with a real decision or tradeoff): 'should I X or Y', 'which option', 'what would you do', 'is this the right move', 'validate this', 'get multiple perspectives', 'I can't decide', 'I'm torn between'. Do NOT trigger on simple yes/no questions, factual lookups, or casual 'should I' without a meaningful tradeoff (e.g. 'should I use markdown' is not a council question). DO trigger when the user presents a genuine decision with stakes, multiple options, and context that suggests they want it pressure-tested from multiple angles."
---

# LLM Council

You ask one AI a question, you get one answer. That answer might be great. It might be mid. You have no way to tell because you only saw one perspective.

The council fixes this. It runs your question through 5 independent advisors, each thinking from a fundamentally different angle. They optionally review each other's work. Then a lightweight summary — produced in the main thread, not by another sub-agent — pulls out each advisor's key point and concrete action, the themes they agree on most, and where they clash.

This is adapted from Andrej Karpathy's LLM Council. He dispatches queries to multiple models, has them peer-review each other anonymously, then a chairman produces the final answer. We do the same fan-out and peer review inside Claude using sub-agents with different thinking lenses — but the chairman sub-agent is replaced with a mechanical summary in the main thread. That saves one sub-agent spawn per session (the single most expensive step after the advisor fan-out) and keeps the user in the driver's seat on the clashes instead of deferring the call to a sixth agent.

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

**The council is expensive. Eight sub-agents per session at full scale
(5 advisor drafts + 3 peer reviews; peer review was downsized from 5
reviewers to 3 because blind-spot detection saturates quickly, and the
chairman agent has been removed entirely in favor of a lightweight
in-thread summary in Step 4).
Prior sessions on this repo repeated the same question 6× in
different wording ("biggest limitation?", "most critical gap?",
"single most critical limitation?" were all the same question). The
user lost agent-time and cycles re-debating settled ground.**

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
The council is expensive — 8 sub-agents at full scale (5 drafts + 3
peer reviews; the chairman sub-agent has been replaced by an in-thread
summary, see Step 4). Use the bucket classification from Step 0 /
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
   to represent your angle as strongly as possible. A summary step will
   surface agreements and clashes later — no chairman will pick a side.

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
  summary surfaces both positions without picking a side; the user
  decides.
- **1-advisor panel (bucket B):** peer review is always skipped. Go
  straight to Step 4.

**Diverge case (only on 5-advisor panel):** if 2+ advisors materially
disagree with the rest, or any advisor flags a "fatal flaw" in the
majority view, **proceed to Step 3**. Peer review is what Karpathy's
method exists for; this is the case it was designed to handle.

Record the convergence decision in the transcript so later Step 0
duplicate-checks can see whether the verdict came from a peer-reviewed
deliberation or a short-circuited unanimous first pass.

### Step 3: Peer Review (3 sub-agents in parallel, ONLY on divergence)

**Run this step only if Step 2.5 determined that the 5-advisor panel
has material disagreement.** Otherwise skip to Step 4.

This is the step that makes the council more than just "ask 5 times." It's the core of Karpathy's insight.

Collect all 5 advisor responses. Anonymize them as Response A through E (randomize which advisor maps to which letter so there's no positional bias).

**Why 3 reviewers, not 5:** blind-spot detection saturates fast — after 3 independent reads of the same 5 anonymized drafts, the 4th and 5th reviewer typically echo what the first 3 already flagged. Dropping from 5 → 3 reviewers saves 2 sub-agents on every diverged bucket-E session at the cost of a small probability that a 4th or 5th reviewer would have surfaced an outlier blind spot. The tradeoff is worth it; in the rare case the summary feels thin, a follow-up turn can spawn additional reviewers on demand.

Spawn 3 new sub-agents. Each reviewer sees all 5 anonymized responses and answers three questions:

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

### Step 4: Write the Council Report (summary + transcript in one file, no sub-agent)

The chairman sub-agent has been removed. Instead of producing a summary
in-thread first and then re-serializing it into a report file, the
main agent writes **one** markdown file directly, with the summary
sections rendered straight into the document. This is the single
final step — no intermediate in-thread summary, no HTML, no PDF, no
separate `council-transcript-*.md`.

**File:** `council-report-[timestamp].md`

**What the summary deliberately does NOT do:**

- It does NOT take a side on clashes. Surfacing both positions is
  enough; the user picks. If the user wants an opinionated call,
  they ask in a follow-up (e.g., "of those clashes, which would you
  lean?") — opting in beats always paying for a chairman agent.
- It does NOT re-argue advisor points or add independent analysis.
  Anything the main agent wants to add goes in a separate follow-up
  turn, not in the report.

**Required structure (top-to-bottom, each section exactly once):**

1. `# Council Report — YYYY-MM-DD [topic slug]` (H1)
2. **Question** — the user's raw question, one short paragraph.
3. **Metadata block** — bucket (A/B/C/D/E), panel size + composition,
   peer-review status (`ran (3 reviewers)` / `skipped (convergence)` /
   `skipped (panel size)`), timestamp.
4. **Per-Advisor Key Point + Action** (H2) — one bullet per advisor,
   in panel order. Each bullet has exactly two parts: the advisor's
   **key point** (one sentence — the core of their draft, not a
   recommendation) and their **action** (one imperative sentence — the
   concrete thing they would do). Extract both from the draft; do NOT
   paraphrase for flavor, embellish, or invent an action the advisor
   did not state. If an advisor's draft contains no concrete action,
   write `Action: none stated.` rather than manufacturing one.

   There is deliberately NO synthesized cross-advisor action list. The
   old "Suggested Actions" callout merged five voices into three
   dependency-ordered bullets, which is a chairman verdict wearing a
   different hat — it decided which advisor won before the user had
   read them. Each advisor's action now stands attributed and
   unranked; the user does the merging.
5. **Where Advisors Agree Most** (H2) — the 2-3 themes that 3+
   advisors (or both, on a 2-advisor panel) converged on
   independently. One line each. If fewer than 2 themes meet the bar,
   list only those. Do not invent agreement to fill space. On a
   1-advisor panel, omit this section entirely.
6. **Where Advisors Clash** (H2) — material disagreements, one line
   each, naming which advisor argued which side. If there is no
   material clash, write exactly `No material clash.` On a 1-advisor
   panel, omit this section entirely.
7. **Blind Spots from Peer Review** (H2) — include ONLY when Step 3
   ran. One line per item flagged by ≥ 2 of the 3 reviewers. Omit
   this section entirely when peer review was skipped.
8. **Load-Bearing Assumptions** (H2) — one line each on the categories
   below. Use `N/A` inline for categories that are truly unrelated;
   do NOT omit the category. MEMORY.md and
   `tests/test_lesson_citations.py` grep these rows to detect
   invalidated priors, so the block must stay mechanical and terse.
   Direct motivation: MEMORY.md §1 "P(1st) ranking is calibrated" row
   carries amendments because O26 found the original +0.37 number was
   shape-encoded; disclosing the assumption makes that failure mode
   `grep`-discoverable instead of accidentally rediscoverable.
   - **Scoring encoding**: e.g., `team-identity (real ESPN)` or `shape-encoded`.
   - **Opponent field**: e.g., `ESPN-national 60/30/10` or `actual 30-person pool` or `N/A`.
   - **RNG / sample count**: e.g., `rng_seed=42, n_sim=5000` or `N/A (deterministic)`.
   - **Year scope**: e.g., `2011–2025 ex. 2020`.
   - **Baseline anchor**: e.g., `vs champ_first_tv` or `vs IID`.
   - **Data sources**: file paths for load-bearing artifacts (must survive the stale-citation gate in `tests/test_lesson_citations.py`).
9. **Framed Question + Prior Art** (H2) — the enriched prompt that
    was sent to every advisor, plus the `Ox`/lessons block that was
    injected.
10. **Advisor Responses** (H2) — each advisor's full 150-250 word
    draft under an H3 (`### The Contrarian`, etc.). Use
    `<details><summary>` only if you want them collapsed by default;
    otherwise plain H3s are fine.
11. **Peer Review** (H2) — either all 3 reviewer outputs, or a single
    line `_Peer review skipped (convergence)._` / `_Peer review
    skipped (panel size = N)._`
12. **Footer** — one-line timestamp + what was counciled.

**Panel-size adjustments:**

| Panel | Sections to include |
|---|---|
| 5-advisor, peer-reviewed (Step 3 ran) | all 12 |
| 5-advisor, converged (Step 2.5 short-circuited) | all except 7; add one line above section 8: `_Convergence: ≥4/5 advisors aligned; peer review skipped — calibrate confidence accordingly._` |
| 2-advisor (bucket A / D) | all except 7; if advisors disagree, section 5 is `_No agreement — both positions stand._` |
| 1-advisor (bucket B) | 1, 2, 3, 4, 8, 9, 10, 12 (omit 5, 6, 7, 11 — no panel to compare, no peer review) |

**Deduplication rule:** every piece of information appears exactly
once. Per-advisor key points and actions live only in section 4.
Advisor drafts live only in section 10. Sections 4-7 (summary)
never re-state advisor drafts verbatim. There is no longer an "agreement /
disagreement" table — section 4 already carries the per-advisor
key points and actions in a scannable form, and a second table rendering the
same data in different syntax was redundant.

**Why writing the summary straight into the file:** producing the
summary as in-thread output first and then re-serializing it into a
`council-report-*.md` meant emitting the same bullets twice (once as
main-thread text, once as file contents). Writing directly to the
file cuts that duplication, shaves tokens from the context window,
and collapses two user-visible "steps" into one.

**What NOT to do:**
- Do not emit any `<style>`, inline `style="..."`, or CSS.
- Do not wrap the document in `<html>` / `<body>`.
- Do not write a second file. `council-transcript-*.md` is deprecated —
  this single `council-report-*.md` replaces both.
- Do not produce the summary as in-thread output before writing the
  file; write directly to the file.
- Do not open the file in a browser; the user reads markdown inline.

**Sub-agent budget (after these optimizations):**

| Path | Sub-agents |
|---|---|
| 5-advisor, peer-reviewed (bucket E, diverged) | 8 (5 drafts + 3 peer reviews) |
| 5-advisor, converged (bucket E, Step 2.5 short-circuit) | 5 (drafts only) |
| 2-advisor (bucket A / D) | 2 |
| 1-advisor (bucket B) | 1 |

Compare to the pre-optimization full scale of 11 (5 drafts + 5 peer
reviews + 1 chairman).

### Step 5: Update the Lessons Index

The single markdown file from Step 4 already contains the full
transcript content (framed question, bucket, panel size, advisor
drafts, peer review, summary). The only remaining housekeeping is
updating the consolidated lessons index.

Recording the panel size and peer-review decision in the report's
metadata block lets future Step 0 dedup-checks tell whether a prior
verdict came from a full peer-reviewed deliberation (8 sub-agents), a
converged-no-review short-circuit (5 sub-agents), or a 2-advisor
re-litigation. That distinction matters when the user says "the
council said X" — a 2-advisor bucket-A verdict has different
evidentiary weight than a full bucket-E verdict.

**A. Append to `COUNCIL_LESSONS.md` if present.** Per that file's §4 Step
4 update rule:

- Append one row to §3 with: next sequential `#`, date, one-line framed
  question, one-line verdict. With no synthesized action list to quote,
  derive the verdict from the strongest agreement theme (section 5); if
  there is no material agreement, write `no consensus — see report`
  rather than elevating one advisor's action into a council verdict.
- If the verdict closes an open `Ox` item in §2 → move the item to §1 and
  leave `[closed <date>]` in §2.
- If the verdict opens a new unresolved item → add a new `Ox` row with
  the next free ID (never reuse, never renumber).
- If the verdict supersedes a §1 lesson → append a superseding bullet and
  mark the old one `[SUPERSEDED <date>]`. Do not rewrite.

**Closure-receipt requirement (CI-gated).** Before tagging an `Ox` row
`[closed <date>]`, the closure narrative MUST cite at least one path
under `scripts/`, `artifacts/`, or `tests/` that exists on disk —
typically the script that produced the evidence, the serialized JSON /
markdown audit, and the lock test that prevents drift. The drift guard
`tests/test_closure_receipts.py` (powered by
`scripts/check_closure_receipts.py`) fails CI on any new closure that
ships without receipts. This exists because §2 O25's first G1 closure
cited spread numbers (0.835/0.315/0.045/0.310) that had no script, no
artifact, and no test on disk — caught only by accident on re-run. Do
NOT add to `BASELINE_UNVERIFIED` to silence a failure; fix the closure
to cite real evidence instead.

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

The top of the file is the scannable verdict (question, per-advisor
key point + action, agreement and clash sections, peer-review blind
spots when applicable, load-bearing assumptions).
The bottom holds the framed question, advisor drafts, and peer review
for later reference. No HTML, no PDF, no second transcript file — the
old `council-report-*.html` + `council-transcript-*.md` split is
deprecated, and the redundant agreement/disagreement table has been
removed now that the per-advisor bottom-line section covers the same
data.

---

## Important Notes

- **Always spawn all advisors in parallel.** Sequential spawning wastes time and lets earlier responses bleed into later ones.
- **Always anonymize for peer review.** If reviewers know which advisor said what, they'll defer to certain thinking styles instead of evaluating on merit.
- **The Step 4 summary is extraction, not judgment.** It does NOT take a side on clashes, it does NOT add independent analysis, and it does NOT re-argue advisor points. If the user wants an opinionated call on a clash, they ask in a follow-up turn — opting into that beats always paying for a chairman agent. If you find yourself writing "the right answer is X" inside the summary, stop: the chairman was deliberately removed so the user owns the call.
- **Don't council trivial questions.** If the user asks something with one right answer, just answer it. The council is for genuine uncertainty where multiple perspectives add value.
- **Keep the report in markdown.** The old HTML report was the slowest step in the pipeline — inline CSS, collapsible `<details>`, visual grids, and then a duplicated markdown transcript all had to be streamed token-by-token. A single markdown file cuts the finalization time and renders cleanly everywhere the user reads it.
