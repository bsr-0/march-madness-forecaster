# Independent Evaluator Audit — Bracket Strategy Backend and 2027 User Value

    Date:        2026-09-09
    Scope:       read-only review of src/optimization, src/simulation, src/evaluation,
                 scripts/mc_pool_backtest.py, the prediction/calibration/data layers,
                 governance artifacts, README/FINDINGS/PROSPECTIVE docs, docs/ web UI, CI
    Stance:      the criticisms a credible, skeptical outside reviewer (quant referee /
                 sports-analytics practitioner) would raise, applied as the judgement
                 criteria for the backend methodology and for the value delivered to a
                 user in March 2027
    Method:      seven independent review passes, each citing file:line; every finding
                 is marked CONFIRMED (code read / command run) or SUSPECTED

No code was changed.

---

## 1. Verdict in one paragraph

The project's central thesis — that prediction accuracy is commoditised and the edge is in
bracket *construction* and *pool-aware selection* — is well argued and the engineering
discipline behind it (walk-forward training, leakage guards, pre-registered freeze,
a candid dead-end ledger) is far above hobby-project norm. But the headline number that
carries the thesis, **"11.2% P(1st) in a 30-person pool over a 15-year backtest"**, would
not survive independent review as stated. It is a simulator-vs-simulator quantity (both the
model bracket and its opponents are scored against a *synthetic* tournament drawn from a
seed model fit on the same years), it is the maximum of a large in-sample sweep, its honest
year-level uncertainty is about ±2.7pp, and the deployed code paths cannot reproduce the
candidate set that was backtested. Separately, the only user-facing deliverable that
currently works is the static web page; the documented CLI fails in every mode. A 2027 user
gets a well-presented, backtested-in-simulation bracket recommendation — that is real value
— but the README's framing overstates what has been demonstrated, and 2027 readiness rests
on a manual, single-person process.

### Scorecard

| Dimension | Grade | One-line reason |
|---|---|---|
| Construction / selection methodology | B | Sound mechanics; objective is optimised and evaluated inside the same synthetic referee |
| Evidence for the headline P(1st) | D | Simulated outcomes, in-sample sweep, no year-level CI, contaminated 2026 included |
| Prediction model integrity | C+ | Game-level PIT is genuinely careful; one confirmed feature leak (roster WARP/RAPM); MC noise never fit |
| Data provenance for 2027 | C | "Pre-tournament" Torvik files are April-2026 date-window reconstructions; no live picks endpoint |
| Governance / pre-registration | B− | Real hash-pinned freeze, but pre-registers no number, no bracket, no failure criterion |
| Documentation honesty | D | README describes a pipeline that does not produce the shipped bracket; 9 of 19 commands and 3 doc links are dead |
| 2027 user value (web page) | B | Usable, shareable, SE-aware display; fixed 30-person ESPN-behaviour WTA pool only |
| 2027 user value (CLI / clone-and-run) | F | `optimize-pool` fails in every mode today |
| Maintainability / bus factor | D | 1 maintainer, 112 scripts, CI red, nightly cron disabled |

---

## 2. What the backend strategies actually are

All under `scripts/mc_pool_backtest.py` unless noted. Scoring is ESPN 10-20-40-80-160-320,
winner-take-all, 30 opponents (`ESPN_SCORING` :129).

| Mode | Mechanism | Status |
|---|---|---|
| `champ_first` / `f4_first` / `e8_first` | pick the top of the bracket from round-probability marginals, then fill down | backtested; dominated by poolaware |
| `region_top_n` | per-region stochastic construction at a "risk" level 0.1–0.9 | the workhorse; a *fixed, unselected* `region_top_n` at risk 0.35 scores 8.4% on its own (`artifacts/backtest_runs/…0830_093733.txt:159`) |
| `exhaustive_champion` | enumerate champions, construct beneath each | candidate family |
| `det_*` | deterministic argmax | 0.00% P(1st); correctly retired |
| `opt_*` / Pareto-leverage (`src/optimization/leverage.py`) | greedy per-game EV × leverage | catastrophic in upset years; correctly retired |
| `meta_sa*` | simulated annealing | 0.7–1.7%; correctly retired |
| **`meta_region_poolaware`** | build ~20–33 candidates per year from ≤5 probability bases × 5 risk levels × 2 constructions (+ forced champions) (:3912-3965), then pick the one with the highest simulated P(1st) against 30 simulated opponents (:4054-4068) | **production claim: 11.2%** |

The probability base that matters is **raw Torvik barthag through log5**
(`src/prediction/pairwise.py:101-112`, backtest :607-645), not the README's 7-feature
logistic regression. The ML pipeline (`src/pipeline/…`, `MonteCarloEngine`) is a separate
stack that never feeds the shipped brackets (`ARCHITECTURE_AUDIT_PREFERENCE_BRACKETS.md`;
`scripts/experiments/build_candidate_artifact.py:769,933`).

Modules an evaluator would expect to be load-bearing but are not exercised by any
backtest: `bracket_portfolio.py`, `dual_submission.py`, `path_protection.py`,
`e8_matchup_scorer.py` (CLI-only); `matchup_vulnerability.py`, `portfolio_diversification.py`
(no callers); the `leverage.py` path-protection gate runs on fabricated 0.6 probabilities
with empty `loser_id` (`leverage.py:960-1148`) and is effectively a no-op.

---

## 3. Findings, ranked by how much they undermine trust

### CRITICAL

**C1. P(1st) is measured against simulated tournaments, not history.** CONFIRMED.
Under the canonical `--team-identity` contract, each repeat draws one tournament from
`seed_pw` with `noise_std=0.16` and scores *both* the model bracket and the opponents
against it (:4328-4360). The realised `actual` result is used only on the legacy shape path
(:4340) and for the `MeanScr` column (:4432). `p_first = (all_ranks == 1.0).mean()` (:4425)
never sees a real outcome. "15-year backtest" therefore means 15 seed layouts × 15 pick
distributions × synthetic tournaments. 2016 Michigan State losing in R64, 2018 UMBC, 2023
FDU never occur. The frozen spec's `p1_definition` does say "seed pairwise referee"; the
README (:45) does not, and a reader will assume "won ~1 in 9 real pools".

**C2. Selector and evaluator are the same distribution, and it is fit on the test years.**
CONFIRMED. `seed_pw = build_seed_probabilities(seeds)` (:2749) uses
`OUTCOME_WINDOW = "recent"` = 2010–2025 (`src/prediction/seed_probabilities.py:33`,
`src/data/seed_pick_model.py:120,131`). The candidate is *chosen* by P(1st) under `seed_pw`
(:4054-4068) and then *scored* by P(1st) under `seed_pw` (:4331), with only the RNG stream
differing. This is in-sample by construction: the metric partly measures "how well did the
selector pick the candidate that wins most often under the referee", and the referee learned
its seed-advancement rates from the years being judged.

**C3. The shipped bracket is not the backtested strategy.** CONFIRMED.
`scripts/generate_poolaware_bracket.py:91-95` builds candidates from `tv` and `mass_avg`
only. The backtest recipe also sweeps `mass_best`, `blend`, `tv_mass80` — and the log shows
`blend_*` selected in 9 of 15 years, `tv_mass80` in 2025. Production cannot produce the
candidate the backtest picked most often. The 2026 submission
(`bracket_2026_submission.json`) was produced by the pool CLI's Torvik/blend path (matches
the report dict at `src/cli/pool_cmds.py:275-288`), not the governance-frozen
`TournamentPipeline`; it records no `prob_mode`, so its probability source cannot be
recovered from the file.

**C3 update, 2026-09-10 — the finding was right but aimed one file behind the product,
and the corrected version is worse.** Fixing it surfaced that there are *three* candidate
recipes, not two, and the one that actually reaches users was never the one named above:

| recipe | bases | risk grid | construction modes | ships to |
|---|---|---|---|---|
| `mc_pool_backtest.py` (measures the ~11%) | tv, mass_avg, mass_best, blend, tv_mass80 | .1/.3/.5/.7/.9 + .3/.5/.7 | region_top_n, exhaustive_champion, forced 1-seed champs | nothing — it is the measurement |
| `generate_poolaware_bracket.py` | was tv, mass_avg → **now all five** | same as backtest | same as backtest | `docs/data/bracket_2026.json`, **which no longer exists** |
| `build_candidate_artifact.py` | torvik, massey_avg, **elo** | .1/.2/.35/.5/.7 | region_top_n only | `candidates_*.json` → `season_*.json` → **the live site** |

`docs/app.js` fetches `seasons.json`, `season_${year}.json` and `training.json` — never
`bracket_2026.json`, which was deleted in `32f860e` ("Remove the UI layer and its contracts
ahead of a rebuild"). So the script the original finding named writes a file nothing reads,
and `deploy-pages.yml:31-40` still lists that dead file (plus `style.css` and
`team_profiles.json`, also gone) among its REQUIRED_FILES — that workflow cannot pass.

The shipping recipe differs from the measured one on every axis: a different rating source
(`elo`, which the backtest's poolaware sweep does not use, in place of `mass_best`/`blend`/
`tv_mass80`), a different risk grid, and no exhaustive-champion or forced-champion
candidates. Some of that is deliberate and defensible — `build_candidate_artifact` builds a
*filterable bank* spanning several worldviews, a different product than "one recommended
bracket," and its docstring argues the case. But the consequence is unchanged and is the
sharper form of C3: **the ~11% P(1st) figure does not describe any bracket the site
displays**, because no bracket on the site is produced by the recipe that number measures.

Fixed here: the backtest and `generate_poolaware_bracket.py` now share one recipe
(`src/optimization/poolaware_recipe.py`, pinned by `tests/test_poolaware_recipe.py`), so
those two cannot drift again. NOT fixed: the live artifact path still uses its own recipe.
Reconciling it is a product decision, not a bug fix — either the bank adopts the backtested
recipe, or the site stops attaching a backtested P(1st) to brackets that recipe never
produced.

### HIGH

**H1. No honest uncertainty is reported; the one CI computed understates by ~3×.**
CONFIRMED. `n_trials = all_ranks.size` (:4430) = brackets × 100 repeats feeds the Wilson CI
(`src/evaluation/testing_budget.py:97`); repeats within a year are not independent
observations of anything real. From the per-year values in the surviving log
(`…20260829_095910.txt`: .01 .11 .14 .05 .15 .21 .07 .08 .06 .09 .11 .15 .14 .15 .04), the
year-level estimate is **10.4% ± 2.7pp (95%, n=15)**. FINDINGS §6e's "SE ≈ 0.79pp" is the
repeat-level figure. "Beats the 4.05% seed baseline" survives; the digit after the decimal
in "11.2%" does not, and the "10.47 vs 11.2 is −0.92 SE" comparison is really ≈ −0.27 SE.

**H2. Garden of forking paths with no untouched holdout.** CONFIRMED. `ALL_MODES` lists 79
strategies (:294-374). Two candidate families were *removed because they lowered aggregate
P(1st)* on the evaluation years (:3967-3983 — 10.93% vs 11.20%; 7.1% vs 11.9%);
`pa_trials` was raised 200→500 because it helped. `PoolHyperparameters` (:423-441)
walk-forwards only `blend_alpha` and `enabled_modes`; the risk grid, 0.8/0.2 blend, trial
budgets, `noise_std=0.16`, pool size are fixed globals with no provenance. The aggregate
includes 2026, which the frozen spec itself classifies as contaminated. The RDoF audit
module (`src/ml/evaluation/rdof_audit.py`) covers only ML-pipeline constants and LOYO Brier;
it has zero references to the pool-strategy search. The FINDINGS permutation test
(p=0.0076) corrected over modes in one run, not over the history of removed candidates.

**H3. Opponent model is synthetic for 11 of 15 years and leaks future behaviour into
early years.** CONFIRMED. Opponents are independent draws (`chalk_noise_std=0.0`,
:2618; `pool_competition.py:239,290`) with per-game pick probability = ratio of two
*marginal* ESPN advancement rates (:365-370). No opponent–outcome coupling, no chalk
clustering, no local-team bias. For 2012 (no ESPN archive) the fallback builds opponents
from `pool_hist_results.json` years **2023–2026** (:2619-2639) — behaviour eleven years in
the future. Real 30-person-pool opponents exist for only 2023–2026, and on those four years
the project's own check is ρ=+0.42, p=0.34, with **2026 = −0.60** (FINDINGS §5b): higher
estimated P(1st) placed *worse*. ESPN pick archives carry no capture timestamp; 2024–2025
are flagged `real_unverified_source` and nothing reads the flag (SUSPECTED contamination).

**H4. The documented CLI does not work in any mode.** CONFIRMED (run). **FIXED 2026-09-09.**
`optimize-pool --mode meta_region_poolaware` (README:46-47) was rejected by argparse —
choices are `auto/torvik/blend/noseed/seed` (`src/cli/pool_cmds.py:1163`); the README example
was already corrected in the earlier headline-claim fix. `--mode torvik` raised
`ModuleNotFoundError: src.prediction.torvik_probabilities` (`pool_cmds.py:858`; module deleted
in commit `44b048f`, 2026-04-21) — restored as a thin wrapper delegating to
`PairwiseProbabilities.from_ratings` (the sanctioned log5 path) and
`scripts.mc_pool_backtest.build_torvik_round_probabilities` (the same function the production
backtest uses), rather than a third independent implementation. `--mode seed` (and every other
mode, transitively, since all of them call `_load_seeds`) crashed on the play-in seed
collision — the 2026-09-06 play-in fix reached the artifact path
(`build_candidate_artifact.resolve_field`) but not the CLI; `pool_cmds._load_seeds` /
`_load_regions` now call the same `resolve_first_four` the backtest and the artifact path use,
via a shared `_resolve_play_ins` helper. Verified against real 2026 data: `--mode torvik`,
`--mode seed`, and `--mode blend` (which also exercises `noseed_model`) each ran to completion
and produced a real bracket with real P(1st)/EV numbers; `--mode auto` (which sweeps all of the
above) also completed. Regression coverage: `tests/test_pool_cmds_play_in.py` (fixture +
one real-2026-data test) and `tests/test_optimize_pool_e2e.py` (runs the actual
`run_optimize_pool` entry point against real 2026 data for `torvik` and `seed`, `~40s` total)
— both marked `integration`/`slow` so the fast suite stays fast. The installed `march-madness`
console script still fails (`No module named 'src'`); `python -m src.main` remains the only
working entry point and is not fixed here — a packaging issue (missing `src` on `sys.path`
outside a repo-root invocation), not a bracket-logic one.

**H5. A confirmed point-in-time leak in two production features.** CONFIRMED.
`total_warp` and `top5_rapm` (members of `SIMPLE_FEATURE_SET`, `src/pipeline/config.py:174-184`)
come from one static per-season roster file that is season-final: `cbbpy_rosters_2024.json`
Purdue `games_played=39`, 2025 Florida `=40` — both include the tournament run. The overlay
is stamped onto every training row, including November games
(`sample_loading.py:541-545`, `_orchestrator.py:519-523`). The post-tournament timestamp
guard (`data_loader.py:359-377`) is warning-only and skipped when `file_year == year`, i.e.
for every historical file. Effect: target leakage in training years and an optimistic 2025
holdout Brier; 2026 inference used a 2026-03-16 snapshot, so there is also train/serve
shift. FINDINGS §4 records excluding roster minutes from the *Bracket Lab* matrix; the same
contamination survives in the pipeline model. (The regular-season features — Elo, win%,
momentum, SOS, tempo, ORB, opp-TO — are PIT-correct: `proprietary_metrics.py:226-244,1485-1486,1681`.)

**Quantified and FIXED 2026-09-10.** The mechanism is explicit in the scraper:
`warp = max(0.0, bpm * minute_share * games_played / 300.0)`
(`cbbpy_rosters.py:405`) is linear in games played, so a team that won six tournament
games carries ~17% more WARP for every player. Measured on the shipped files, the
correlation between a team's roster `games_played` and the tournament rounds it actually
won is **+0.49 to +0.83 in every season 2011–2025** (champions carry 4–6.5 more games than
R64 losers, i.e. exactly their tournament wins), against **−0.05 for 2026**, whose snapshot
is genuinely mid-February. The team with the most games in each file is that season's
champion or runner-up — 2008 Kansas, 2011 UConn, 2017 UNC, 2022 Kansas. `diff_total_warp`
is the feature `SIMPLE_FEATURE_SET` annotates as the "largest coefficient".

Fixed at the loader rather than the feature list, because both production configs set
`enable_feature_selection: true` and therefore never read `SIMPLE_FEATURE_SET` — editing
that list would have left the contaminated values available to the learned selector.
`load_roster_overlay` now detects contamination correctly (the old check bailed out when
`file_year == year`, which is the contaminated case and true of every file on disk), drops
the overlay with an error-level log, and raises `RosterContaminationError` under
`strict_leakage_mode` — which both production configs set. Verified against the repo's own
data: 2011–2025 drop, 2026 is retained. Pinned by `tests/test_roster_contamination_guard.py`.

**Rebuilt 2026-09-10 (the "rebuild" option, chosen).** The FINDINGS note saying the rebuild
was "blocked until game-level box scores land" was stale: `boxscores_{2008..2026}.json` had
landed on 2026-08-25..27 — ~5–6k dated games per season, every one before its tournament
cutoff. `scripts/build_boxscore_rosters.py` (`src/data/features/boxscore_rosters.py`)
re-aggregates those games through the *same* `CBBpyRosterScraper._build_payload` formulas, so
the rebuilt feature differs from the old one only by the games included, and bridges ESPN
slugs onto canonical ids via `resolve_cbbpy_bridge` against the full D1 universe (94–97%
bridged; 62–64 of 64 tournament teams matched per season). Result, r(roster games_played,
tournament rounds won):

| | cbbpy (shipped) | rebuilt |
|---|---|---|
| 2011–2025, range | +0.49 … +0.83 | −0.04 … +0.30 |
| 2011–2025, mean | ≈ +0.72 | ≈ +0.12 |
| 2026 (genuine Feb snapshot; control) | −0.05 | +0.10 |

The rebuilt 2026 file — which cannot contain tournament games — scores the same +0.10 as the
rebuilt historical seasons, so what remains is conference-tournament depth (real, pre-cutoff
information), not the leak. Purdue 2024: 39 → 32 games, WARP 1.26 → 0.70. The rebuilt files
carry `data_type: pre_tournament_rosters` and `max_game_date`, which the guard checks
instead of the scrape timestamp (as `data_as_of` is for Torvik); every roster call site
(`sample_loading`, `seed_baseline_loyo`, `backtest_harness`, `roster_adj_probabilities`,
`sweep_training_window`) now prefers `rosters_boxscore_{year}.json`. Pinned by
`tests/test_boxscore_rosters_rebuild.py`, including a real-data test that the 2024 file's
correlation is under 0.25. The train/serve skew below is therefore closed for every season
with box scores (2008–2026); `is_transfer`/`eligibility_year` remain the constants cbbpy
always shipped (False / 1), so overlay slots built from them are inert, as before.

**The consequence is a decision someone has to make, and is deliberately left open.**
Dropping the overlay for contaminated seasons means training rows carry no roster features
while a clean 2026/2027 snapshot still would — a train/serve skew of exactly the shape
FINDINGS §6c describes, where a feature is constant in training and real at inference. That
is better than leaking but is not a resting state. Under `strict_leakage_mode` the pipeline
now raises instead of skewing, which forces the choice rather than making it silently. The
honest options are to rebuild roster features from game-level box scores
(`src/data/scrapers/espn_boxscore.py`, which FINDINGS §6d establishes has full historical
coverage) or to drop them from both training and inference until that exists.

**H6. Brackets are constructed for one pool size and scored against another.** CONFIRMED
(found 2026-09-10 while re-measuring the headline). **FIXED 2026-09-10** — see the closing
paragraph of this entry. `_run_one_year` resolves the real field
size per season and stores it — `pool_size = year_n_opponents + 1`
(`mc_pool_backtest.py:2813`) — but that variable is used only for the P(top5%)/P(top25%)
thresholds (`:4426-4427`). Every bracket-construction call site instead passes the raw CLI
value: `pool_size=n_opponents` (`:3779, 3844, 3862, 3914, 4004, 4114, 4135, 4176, 4230,
4253`). So for any season with real pool history the bracket is *built* for the CLI's
assumed pool and then *scored* against a field of a different size (2023–2026 real sizes are
19, 26, 33 and 31 entries). The same line is also off by one against its own callee's
semantics: `construct_bracket`'s `pool_size` means total entries, and
`generate_poolaware_bracket.py` correctly passes `n_opponents + 1`, while the backtest
passes `n_opponents`.

**Currently latent, and the reason matters.** `_make_ev_scorer` applies `pool_factor` only
when `pool_size > 50` (`bracket_construction.py:204`); below that it is exactly 1.0. Every
real pool in the data (19–33) and the canonical `--n-opponents 29` all sit under that
threshold, so construction is bit-identical whether the right or wrong size is passed, and
**the 11.9% headline is unaffected**. The bug bites only when `--n-opponents` is large —
which was the default (999) until 2026-09-10, when it was changed to 29 (a 30-person pool)
for this reason. Measured directly: at the old default the 2023 sweep dedups to 12
unique candidates and selects `mass_avg_region_risk=0.1`; at `--n-opponents 29` the same
season yields 28 candidates. A run at the default therefore constructs heavily contrarian
brackets for a 1000-person field and scores them against a 19-person one. This is the
second defect found in this file that is invisible at the pool sizes anyone uses and
catastrophic at the default — see also the pool-size provenance gap in the run header.

**The selection half is worse than the construction half, and is not latent.** Candidate
*selection* has the same defect: the three `draw_selection_trials` calls in `_run_one_year`
also took the raw CLI value, so the selector estimated each candidate's P(1st) against a
29-opponent field and the result was then scored in the real one. Unlike construction, this
has no `pool_factor` threshold protecting it — simulating a 29-opponent field when the pool
holds 18 changes the P(1st) estimates directly, and therefore changes which candidate is
selected, at the canonical settings.

**Fix.** All ten `construct_bracket` calls now take `pool_size=pool_size` (total entries)
and all three `draw_selection_trials` calls take `n_opponents=year_n_opponents` (opponents
only); both derive from the season's resolved field rather than the CLI fallback.
`tests/test_field_size_threading.py` walks the AST of `_run_one_year` and fails on any call
that reverts to the raw parameter — mutation-tested by reverting one site, which the guard
catches by line number. A static guard is the right shape here precisely because a
behavioural test at a realistic pool size cannot see the construction half of the bug.
Note `generate_poolaware_bracket.py` had this right all along (`pool_size=n_opponents + 1`
against a `n_opponents` already resolved to the real group size); the backtest was the
outlier.

### MEDIUM

**H7. The ML pipeline cannot load data at all: a deleted package is still imported.**
CONFIRMED (found 2026-09-10 while fixing H5). `src/pipeline/stages/data_loader.py:37`
imports `src.conference_tournament.data_enrichment`; that package was deleted in commit
`44b048f` (2026-04-21, message "upd") and never restored, so `data_loader` and
`sample_loading` both raise `ModuleNotFoundError` on import. `pipeline_runner` imports
`data_loader` lazily inside its delegation functions (`:703-766`), so the failure surfaces
at runtime on the first data load rather than at import — which is why `src.main` and
`tournament_pipeline` still import cleanly and the breakage has gone unnoticed.

The practical meaning: **the ML training/production pipeline the README describes under
"How it works" has been unable to run since 2026-04-21.** This is consistent with every
other finding here — the shipped brackets come from `mc_pool_backtest.py` and
`build_candidate_artifact.py`, neither of which touches this code path. It also explains why
H5's leak, though real, is currently latent: it contaminates a pipeline nothing runs.

Note the same commit also deleted `src/prediction/torvik_probabilities.py` (H4) and
`scripts/compute_pretournament_barthag.py` (M1). One unreviewed "upd" commit broke at least
three separate paths, and each was found only by trying to execute them.

**H7 update, 2026-09-10 — it is a family of 37, and the ML pipeline now runs.** A strict
scan (module *and* attribute must resolve; the first pass wrongly accepted a parent package)
finds **37 imports across `src/` and `scripts/` that point at modules deleted in `44b048f`**,
against 95 of that commit's 124 deleted files never restored. Most are lazy or `try`-guarded
and only fail when an optional feature runs. Four were not, and each took a whole subsystem
down with it: `data_loader.py` (module level → no data loading at all),
`baseline_training/_embeddings.py` (module level → no training),
`src/ml/training/__init__.py` (eager re-export of two deleted feature modules → the package
holding `symmetric_augment`, which every training run needs, was unimportable), and
`src/data/ingestion/collector.py` + `historical_pipeline.py` (five deleted scraper names →
**the README's `ingest` / `ingest-historical` commands, the first step of the March 2027
runbook, have been unimportable since April**). `src/espn/__init__.py` had the same shape.

Fixes, by kind: load-bearing data structures and guards restored verbatim from `44b048f~1`
(`ml/gnn/schedule_graph.py`, `ml/transformer/game_sequence.py`,
`ml/evaluation/evaluation_integrity.py` — the last is the `YearSplitPolicy` leakage guard);
abandoned *feature* modules made optional at their package `__init__` or call site with an
error naming the commit, rather than resurrected. `tests/test_src_packages_import.py` imports
every one of the 28 `src` packages from the filesystem (not `pkgutil`, which hides everything
beneath a broken package) and fails on any missing `src` module. All 28 pass.

**Two more defects the first end-to-end run then exposed, both in the harness, both of which
had been silently converting model failures into seed-baseline scores:**
1. *No `teams_{year}.json` exists for any season.* It is a virtual path that
   `DataLoader.load_teams_from_json` redirects into `tournament_context_{year}.json["teams"]`
   — but the pre-run validator checked the literal string, so every LOYO fold failed
   validation before training. Fixed with `DataLoader.resolve_teams_json_path`, used by both.
2. *The harness substituted the seed baseline on any failure and reported it in the model's
   Brier column* with only a log warning. A fresh run reproduces
   `configs/backtest_baseline.json`'s 2025 entry (0.141963) to four decimals from the seed
   fallback, so the committed regression baseline is at least partly the seed model scoring
   itself. Fallback is now off by default (`--allow-seed-fallback` to opt in), every year
   carries `per_year_source`, fallback years are marked and excluded from the mean and the
   gate, and `--walk-forward` trains only on earlier seasons (the default LOYO trains on later
   ones too). `tests/test_backtest_harness_provenance.py`.

**And one in feature assembly:** the roster overlay stamped values by hardcoded index from a
retired 71-wide layout (`TEAM_FEATURE_DIM` is 56). Indices 69/70 raised `IndexError`; the
in-range ones were wrong too — `avg_experience`/`bench_depth` off by one (overwriting
`xp_per_poss`), `top5_minutes_share` written into `injury_risk`. A March 2026 commit had
renumbered 74/75 → 69/70 to stop an earlier `IndexError`. Three copies existed
(`data_loader`, `sample_loading`, `_orchestrator`). Now resolved by feature name against
`TeamFeatures.get_feature_names()` in one place, with a static test that fails on any literal
index at an overlay site (`tests/test_roster_overlay_indices.py`).

Not fixed here: resolving it means either restoring the package or removing the call site
at `data_loader.py:858`, and the FINDINGS §4 note that Four Factors were consolidated into
`torvik_{year}.json` in 2026-08 suggests the enrichment may now be redundant — but that is a
judgment about intent, not a mechanical fix.

**H8. The served team vector clipped every Elo rating to 1000, so the production model
predicted ~0.5 for everything.** CONFIRMED (found 2026-09-11, the first time a walk-forward
fold ran end to end). `TeamFeatures.to_vector()` ended with `np.clip(result, -1000.0,
1000.0)` — a "clearly broken data" guard added 2026-02-19 (`db15a57`). Elo ratings live on a
1000–2200 scale (1278–2128 across the 2024 field), so every team was served
`elo_rating = 1000.0` exactly and `diff_elo_rating` was identically zero for every matchup.
Perturbing `elo_rating` on a `TeamFeatures` moved no vector slot at all. Training vectors come
from `metrics_to_team_vector()`, which never clipped, so the model was fit on Elo and served
without it — the FINDINGS §6c failure shape, on the feature that matters most: on the 2024
walk-forward fold the fitted logit carries coefficient 1.12 on `diff_elo_rating` (training sd
250 points) and nothing else above 0.12. Consequences, measured: every prediction within ±0.07
of 0.5 (UConn–Stetson 0.52), Brier 0.246 on 63 main-draw games — worse than the seed
baseline's 0.177 on the same rows and far behind the site's fitted model at 0.137 — and the
Monte Carlo giving 1-seeds 9.6% of titles. The March 2026 production run
(`run-production-2026`) used this same `_raw_fusion_probability → to_vector` path, so its
probabilities were flat too; it shipped nothing user-facing only because the shipped brackets
never came from this pipeline (C3). Fixed by removing the clip; `tests/test_team_vector_fidelity.py`
asserts vector *content* (Elo round-trips at real magnitudes; each field moves exactly its
slot) where the module previously asserted only its length.

**Corollary for the recorded ML numbers.** `artifacts/backtest_result_temperature.json`
(2026-05-06) evaluates 18 folds in 47.8 seconds with per-game predictions drawn from the seed
table (values of exactly 1.000, 0.967, 0.033) and a 2025 Brier equal to the seed baseline's:
it is the seed fallback end to end. `artifacts/backtest_result.json` (2026-04-10, 174,167 s)
is the 48-hour run the README's baseline descends from: 12 of its 17 per-year Briers equal the
seed-fallback file's to four decimals (2008–2017, 2025), so the model was actually scored in at
most five seasons (2018, 2021–2024) — and each of those ran after the 2026-02-19 clip, with
Elo amputated. No artifact in the repository contains a Brier produced by this model with its
features intact; the first such number is the one measured below.

**M1. "Pre-tournament" Torvik ratings are post-hoc reconstructions.** CONFIRMED / SUSPECTED.
All 22 `torvik_{2005..2026}.json` files carry `scraped_at: 2026-04-06` — after the 2026
title game. They are `trank.php?begin=…&end=cutoff` date-window recomputes
(`torvik.py:566-587`), not archived Selection-Sunday pages. `rescrape_pretournament_torvik.py:694-696`
itself warns Torvik "revises ratings for past windows … the drift is silent", and :410-440
falls back to the full-season `{year}_team_results.json` URL when the filtered request
returns empty, still labelled `pre_tournament`. The `_validate_pretournament` guard
(:560-568; `noseed_model.py:33-46`) checks the label string, not content. FINDINGS §4:427-430
still says barthag is "locally computed" by `scripts/compute_pretournament_barthag.py`,
which was deleted in the same April commit.

**M2. Monte Carlo noise parameters were never fit.** CONFIRMED.
`artifacts/mc_calibration_2026.json`: "Placeholder calibration using production config
defaults" (`noise_std 0.16`, `regional_correlation 0.05`). `calibrate_mc_parameters`
(`mc_calibration.py:333`) has zero callers. `pipeline_runner.py:613-628` loads `best_params`
without checking the note; the 2027 file named in `production_2027.json` does not exist and
silently yields None. The backtest's `noise_std=0.16` (:4333) is the same unfit constant.
Pipeline MC otherwise draws independent Bernoulli with i.i.d. per-game logit noise
(`monte_carlo.py:294-301`) — a calibration knob, not an uncertainty model; injury shock and
regional correlation are disabled in production (`simulation.py:135`, `config.py:636`).
Joint per-sim outcomes are discarded (`monte_carlo.py:459-495`), so the pipeline stack
cannot do pool scoring at all — which is why a second, non-shared simulator exists.

**M3. Calibration regime changes silently between 2026 and 2027.** CONFIRMED.
2026's config never sets `calibration_years`, so the default 2008–2025 fires (~1,000
tournament games) while logs claim "holdout-year OOS by default"
(`stages/calibration.py:293-296`). 2027 sets `calibration_years: [2026]` — one tournament,
~63 games, padded to the 80-sample floor with current-year *regular-season* games. Flags
`enable_round_weighted_calibration`, `enable_goto_conversion`, `seed_prior_weight`,
`enable_vegas_calibration_anchor` are set `true` but live only on
`predict_probability_experimental` (`tournament_pipeline.py:858-907`); production is raw →
temperature → shrink-to-0.5 → clip. When the bootstrap CI for T contains 1.0 the calibrator
silently becomes identity (`stages/calibration.py:437-468`). No Python test exercises
`_fit_calibration`'s split.

**M3 update, 2026-09-10 — historical calibration rows were scaled twice in production.**
Found by running the harness end-to-end. The historical-tournament calibration loader
(`stages/calibration.py::_load_tournament_cal_year`) called `scaler.transform` on each
year's rows and then `predict_proba_batch`, whose `_scale_batch` applies the model's fixed
feature indices and scaler again. On the fixed-feature path this raised
("X has 60 features, but StandardScaler is expecting 9"), every historical year was skipped,
and the fit fell to ~47 current-year rows — which is why every harness fold died in
calibration once it could reach it. On the learned-selector path — `enable_feature_selection:
true` in both production configs — it did not raise: the rows were standardized **twice**, so
the 2026 temperature (fit on 2008–2025 tournaments, per the M3 finding above) was fit on
distorted probabilities and nobody could have seen it. Fixed by removing the manual step;
pinned by `tests/test_calibration_loader_scaling.py` on a real fitted model. The 2026
production calibration should be re-fit before that pipeline is used for anything.

**M4. The pre-registration pre-registers nothing falsifiable.** CONFIRMED.
`prospective_2027_v2_scoped.json` + `tests/test_frozen_2027_spec.py` freeze *methodology*
(hash, features, scoring, pool size 30, 2000 trials) and that is genuinely valuable. But no
number, no bracket, no pool, and no failure criterion is stated. A single 30-person pool is
one Bernoulli draw at p≈0.1; 2027 cannot confirm or refute 11.2% and the document should
say so. PROSPECTIVE_2027_v2.md:53-58 cites `configs/frozen/product_v3.json`, `docs/build.js`,
`tests/test_spec_boundary.py` — none exist (`frozen_spec.py:71` still points at the missing
path). Both freeze artifacts were produced from dirty trees (`git_dirty: true`);
`artifacts/pipeline_freeze_2026.json` is labelled `pre-registration` but dated 2026-04-28,
after the tournament; `production_2027.json` sets `require_freeze_file: false`.

**M5. CI is red and the nightly gate is off.** CONFIRMED.
Latest `CI Pipeline` run (2026-09-09, 19h17m) failed: `Full Test Suite` on
`FileNotFoundError` for gitignored `artifacts/candidates/candidates_2024.json`
(`tests/test_ui_filter_payload.py`) plus `test_baseline_evaluation.py` and
`test_schemas.py`; `Browser Model` on "shipped season payload matches the training path
exactly — differs by 0.8252 at 2018 texas_southern.massey_avg_rank" (a train/serve parity
failure in a *shipped* payload). `nightly-validation.yml` has its cron commented out; its
last scheduled runs (Aug 6–8) failed. README:91 says the backtest "runs in CI nightly".
Local `pytest --co` fails at import (`pytest_asyncio` incompatibility). No test pins any
headline number; backtest tests check plumbing equivalence, not scoring correctness against
a hand-computed bracket.

**M6. Structural gaps versus what a sophisticated pool player expects.** CONFIRMED.
Winner-take-all only on the backtest path; `payout_structure` exists in `pool_optimizer.py:40`
but only feeds a manifest. No upset bonus, seed-weighted or round-multiplier scoring
variants; no multi-entry hedging or Kelly-style sizing anywhere backtested; `pool_factor`
activates only above 50 entries (`bracket_construction.py:204`); no pool-size, payout or
scoring input in the UI; no live update after R64; no ingestion of the user's own pool
beyond one hand-scraped ESPN group in a bespoke JSON schema. Tie handling differs between
selection (`>=` counts a tie as a win, :2372) and evaluation (half credit, :4360).

### LOW

**L1. Silent-default surfaces of the same class as the FINDINGS §6c skew bug.**
`barthag.get(t, 0.5)` (:799), seed fallback `1 - seed*0.04` (:631-634),
`model_round_probs…get(round, 0.5)` (:2433), `pub_pct = 0.5` (:3491), `except Exception:
pass` around candidate families (:3738, 3748, 3840, 3909 — a vanished family changes the
sweep with no log line; the varying "best of 20–33" per year is consistent with this),
`feature_engineering.py:1025-1056` defaults (win_pct 0.5, Elo 1500, barthag 0.5), whole
training years skipped on any exception (`_data.py:220-222`).

**L2. Team-name resolution collisions (live probe).** `"North Carolina St."` →
`north_carolina`; `"Saint Mary's (MD)"` → `saint_mary_s__ca`; `loyola_md` vs `loyola__md`
coexist. Torvik join tolerates up to 20% of bracket teams missing before failing
(`data_loader.py:1292,1297`). `configs/team_aliases.json` (v2026.3) has no 2027 entries.

**L3. 2027 hardcoding and dead dependencies.** Kaggle slug list stops at 2026
(`kaggle_downloader.py:25-32`); year defaults of 2025/2026 in `game_utils.py:212`,
`_loyo.py:161`, `scrape_cmds.py:74`, `_helpers.py:174`; ESPN picks scraper tries five
undocumented URLs whose payload shape it cannot parse (`espn_picks.py:121-125,210`) — the
realistic 2027 path is a hand-made `public_picks_2027.json`. Orphan root files
(`barttorvik_2026.csv` is 32 conference-aggregate rows; the 2020 CSV is full-season) and 15
April `pool_report_*.json` files are read by nothing.

**L4. Documentation rot.** README references `CLAUDE.md` three times (:45, :58, :142);
it does not exist. Nine of nineteen documented commands (`sota`, `pre-tournament-check`,
`validate-vs-market`, `freeze-pipeline`, `verify-freeze`, `monitor`, `snapshot`,
`list-snapshots`, `restore-snapshot`) do not exist. README's "7 features" is a 9-item
`SIMPLE_FEATURE_SET` with `enable_feature_selection: true`, so the production set is
learned, not the listed seven. `ARCHITECTURE_AUDIT…` line citations are stale.

---

## 4. Value to a user in March 2027

**Non-technical pool participant via the GitHub Pages site.** This is the real product and
it is good: a year strip 2010–2027, two precomputed strategy cards ("maximise chance of
winning" — P(1st) 10%, 874 pts for 2026 — and "maximise expected points"), an in-browser
fitted ridge model, a filter panel over ~1,200 candidates (champion, 1-seed count, F4 depth,
six shape predicates, rating source), near-tie alternates within one SE, URL-hash sharing,
copy/print export, past seasons graded against actuals, whole-percent P(1st) with a
mandatory 30-entry-ESPN-pool disclosure. Time to a bracket ≈ 2 minutes — *if* the
maintainer rebuilds on Selection Sunday. Today `season_2027.json` is a `not_started` stub
and nothing automated will change that: `build_ui_payload.py:282-286` flips it only when a
gitignored `candidates_2027.json` exists; `generate-web-data.yml` references files that no
longer exist in `docs/`; only `deploy-docs-on-push.yml` works. The March sequence
(seeds → 12 play-ins resolved → picks captured by 2027-03-18 12:00 ET → artifact → payload →
deploy, all before ~12:15 tip) exists only in commit messages and script docstrings. What
the user cannot do: set pool size, payout, scoring rules, number of entries, or feed in
their own pool.

**Technical friend cloning the repo.** Cannot get a bracket from the CLI (H4). Reconstructing
the `scripts/experiments/` artifact path is days of work with no runbook.

**Maintainer.** Feasible. Bus factor is one: 1,608 commits (913 in March 2026, 18 in
September), authors Claude 824 / bsr-0 441 / Ben Rosen 282; 112 scripts; three overlapping
PROSPECTIVE docs; a 73 KB FINDINGS.md.

**Against the 2027 landscape** (KenPom/Torvik bracket odds, ESPN "who picked whom",
PoolGenius-style contrarian optimisers): the genuine differentiators are (a) a walk-forward
15-season evaluation of a *P(1st)* objective with SE-aware display and a hash-pinned
methodology freeze — rare among free tools — and (b) filter-by-belief over a diverse
candidate bank. What commercial tools offer that this does not: pool size / payout / scoring
configuration, multi-entry portfolios, ingestion of the user's actual pool, and live
re-optimisation after R64. An evaluator would conclude the project is a strong research
artifact and a decent single-pool recommender, not yet a general pool tool.

---

## 5. What survives scrutiny (be fair)

- The construction/selection-over-prediction thesis is supported *within the simulator*:
  seed 4.05% vs poolaware ≈10% is many SE apart even at year-level uncertainty, and a fixed
  unselected `region_top_n` already gets 8.4%, so the mechanism is "submit a deterministic,
  chalk-leaning, diverse-enough bracket" rather than luck.
- Game-level point-in-time discipline in the regular-season features is genuinely careful
  (`LeakageError` on cutoff after tournament start, Elo snapshot before as-of date,
  Massey day-bounded, tournament games excluded from training).
- The pairwise-probability contract (`src/prediction/pairwise.py`, AST-scanned by test) fixed
  a real marginal→pairwise error and the project re-baselined honestly afterwards.
- The dead-end ledger, the "if a change is smaller than its bootstrap CI it is not a
  finding" stopping rule, and the play-in resolution fix are exactly the habits reviewers
  ask for and rarely see.
- The methodology freeze with CI drift gate is real, and the CHECKPOINT documents record
  deferred decisions before outcomes exist.

---

## 6. Recommendations, in priority order

**Before making any external claim**
1. ~~Restate the headline~~ **DONE 2026-09-09, re-measured 2026-09-10.** README now states
   **12.0% (95% CI 8.6–15.4%), n=14, 2011–2025, vs seed 4.0%** — measured on current code,
   not quoted from a stale log — with the simulated-tournament / simulated-opponent /
   season-level-CI / pool-size qualifiers spelled out; `CONTAMINATED_EVAL_YEARS` strips 2026 from every aggregate,
   paired test, and returned result in `mc_pool_backtest.py` (2026 still *runs*, for
   integration purposes, but never scores).
2. ~~Report P(1st) with a year-level CI~~ **DONE 2026-09-09.** `print_aggregate_block` now
   prints a t-interval over seasons (2.16σ at n=14, not the previous 1.96 normal
   approximation, which understated it by ~10%) next to every P(1st); `_mean_and_ci95`
   fixed the same way for `run_experiment.py`'s CI. Both the reporting closure and the
   season-CI math are covered by `tests/test_aggregate_season_ci.py`.
3. ~~Add the one number that would actually support the thesis~~ **DONE 2026-09-09.**
   `scripts/real_pool_placement.py` scores the production `meta_region_poolaware` pick
   against the real tournament result and ranks it against the real pool for 2023–2026 (the
   only years real pool data exists), reusing `--save-brackets`'s existing
   `score_brackets_team_identity` output rather than reimplementing scoring. Result (as
   re-measured after the H6 fix): **0 of 4 finished 1st, 0 of 4 finished top 3**
   (18th/18, 4th/25, 10th/32, 12th/30) — it beats a
   fairly-computed mean-of-50 `seed` baseline in every year, but has not won a real pool.
   n=4, not a rate; full table in `artifacts/real_pool_placement/placement_2023_2026.txt`
   and README's "What the backtest number means". Building this also found and fixed a real
   crash: `--save-brackets` indexed `model_brackets[m]` with `range(n_model)` (the
   50-bracket stochastic default) instead of `range(model_brackets.shape[0])`, so it raised
   `IndexError` on every meta mode (1 bracket) — this is very likely why the checked-in
   `artifacts/backtest_brackets/*.json` files were stale from April and never covered
   `meta_region_poolaware`. Fixed and pinned by `tests/test_save_brackets_meta_mode.py`.

**Before March 2027**
4. ~~Fix or delete the CLI~~ **DONE 2026-09-09.** See H4 above.
5. ~~Make `generate_poolaware_bracket.py` sweep the same bases as the backtest recipe~~
   **PARTLY DONE 2026-09-10.** The recipe now has one definition
   (`src/optimization/poolaware_recipe.py`) that both the backtest and
   `generate_poolaware_bracket.py` import, so those two cannot drift again; the shipped
   script gained the three bases it was missing (`mass_best`, `blend`, `tv_mass80`) and now
   displays the winning base's own probabilities instead of always torvik's. But see the
   C3 update: the script this recommendation named is not the live path, and the live path
   (`build_candidate_artifact.py`) still uses a third, different recipe. **Still open:**
   decide whether the shipped candidate bank adopts the backtested recipe or the site stops
   quoting a backtested P(1st) for brackets that recipe never produced.
6. ~~Remove `total_warp`/`top5_rapm` from training; make the roster timestamp guard a hard
   error~~ **DONE 2026-09-10** (guard half). See H5: the guard was inverted and warning-only;
   it now drops the overlay and raises under `strict_leakage_mode`. **Still open:** rebuild
   roster features from game-level box scores, or drop them from inference too — today a
   clean 2026/2027 snapshot would be served to a model trained without them.
6b. **Resolve H7 before anything above matters for the ML path.** `data_loader` imports a
   package deleted in April, so the pipeline cannot load data at runtime.
7. Either fit `mc_calibration` or delete the placeholder and hard-code the constant with a
   comment saying it is unfit; create the 2027 file or make its absence loud.
8. Turn the March runbook into a script or a checked-in document; wire
   `candidates_2027.json` → payload → deploy into one workflow.
9. Get CI green: commit or fixture the candidate artifacts the tests need, fix the
   `pytest_asyncio` pin, re-enable the nightly cron or delete the README claim.
10. Add to PROSPECTIVE_2027 a sentence stating what April 2027 can and cannot conclude at
    n=1, and fix the three dangling references.

14. **Reconcile or retire the third recipe.** See the C3 update: the live artifact path
    (`build_candidate_artifact.py`) still builds its bank from a different base set, risk
    grid and construction mode than the measured strategy, so the headline figure describes
    no bracket the site shows. Also `deploy-pages.yml:31-40` still lists three files deleted
    with the old UI (`bracket_2026.json`, `style.css`, `team_profiles.json`) among its
    REQUIRED_FILES, so that workflow cannot pass, and `generate_poolaware_bracket.py` writes
    a file nothing reads.
15. **Unify the opponent-count default.** `run_experiment.py` defaults to 30 opponents (a
    31-person pool) while the canonical contract is 29 (30-person), so its numbers are not
    directly comparable to the headline.

**Structural (2028)**
11. Evaluate against an *independent* referee (e.g. market-implied or Torvik pairwise, not
    the seed model used for selection) and add real-outcome placement as a co-primary metric.
12. Bring the pool-strategy search under the RDoF audit; keep one never-touched holdout year.
13. Opponent model with chalk clustering / correlated picks, pool-size and payout inputs,
    multi-entry support — the features that separate a recommender from a pool tool.

---

## Appendix — documentation vs code

| Claim | Where | Reality |
|---|---|---|
| "7 domain features, single logistic regression … temperature scaling … 50k MC → optimization" | README:7-16 | Shipped brackets use Torvik barthag log5 + a 10k-sim bracket MC; `n_sims=50000` appears nowhere (default 10,000) |
| "current baseline 11.2% P(1st), 15-year backtest — see CLAUDE.md" | README:45 | Simulated-tournament metric; CLAUDE.md absent. **Re-measured 2026-09-10 on current code: 11.9%, 95% CI 8.6–15.2%, n=14 (2011–2025), vs seed 4.0%.** The 11.2% and every other pre-`b73d351` figure was computed on brackets with unresolved play-in slots — void, not superseded |
| `optimize-pool --mode meta_region_poolaware` | README:46-47 | Not a CLI mode; CLI fails in all modes |
| "LOYO backtest … runs in CI nightly" | README:91 | Cron commented out; last scheduled runs failed |
| "barthag is locally computed … guarded by `_validate_pretournament()`" | FINDINGS §4 | Script deleted 2026-04-21; barthag is scraped; guard checks a label string |
| "holdout-year OOS by default" | `stages/calibration.py:293-296` | 2026 fit on 2008–2025 by default |
| `product_v3.json`, `docs/build.js`, `test_spec_boundary.py` | PROSPECTIVE_2027_v2.md:53-58 | None exist |
| `freeze_type: pre-registration` | `artifacts/pipeline_freeze_2026.json` | Dated 2026-04-28, `git_dirty: true` |
| "SE ≈ 0.79pp" on poolaware P(1st) | FINDINGS §6e | Repeat-level; year-level SE ≈ 1.4pp |
