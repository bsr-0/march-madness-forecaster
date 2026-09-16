# PROSPECTIVE 2027 — methodology freeze 2027.v3 (post-audit)

spec_version: **2027.v3** · freeze_date: **2026-09-15** · spec file:
`configs/frozen/prospective_2027_v3_audited.json` · spec_hash:
`183f90796d527f0f254cc087c349bd72852c43865676d5d175d79a122e89974a` · code state: commit `75b97e4` plus this freeze.
Supersedes 2027.v2 (`configs/frozen/prospective_2027_v2_scoped.json`, hash
`c7d1c67663601bd4…`), which stays on disk immutable; v1 likewise.
Drift gate: `tests/test_frozen_2027_spec.py` compares the live code to this file.

## 1. Production methodology (what generates the 2027 brackets)

- **Probability model, shipped product (Path 2):** Torvik `barthag` (pre-tournament
  snapshot, provenance-gated) plus Massey composite and Elo converted to
  barthag-equivalents → Log5 pairwise → 50,000 simulated tournaments per source,
  no noise → scenario bank. EV = Σ_R pts_R · P(picked team wins R) on Torvik
  marginals. Stratified sample of ~3,000 candidates (champion quotas, floor 8, EV
  deciles) + constructed/shipped brackets, strict winner-set encoding.
- **Live fitted-model tab (Path 1):** 11-feature ridge on margin, no intercept,
  λ = 1 per 1,000 rows, trained on tournament games strictly before the displayed
  season; Student-t link (a, ν) fitted on prior out-of-sample rows only and shrunk
  n/(n+63) — identical in `docs/fit.js` and `src/prediction/pit_production_model.py`.
- **Training-data boundary:** everything strictly before the season being
  predicted (walk-forward). Seed table `as_of=Y`; noseed marginals `as_of=Y`,
  window "recent"; pit rows y < Y; market referees games before March 15 of Y.
- **Tournament simulation:** `simulate_tournament_outcomes` on the season's real
  positional tree; independent Bernoulli per game with tree dependence; referee
  logit noise 0.16 (measured: a ≤0.13 pp shrink toward 0.5, no variance component);
  cap [0.01, 0.99].
- **Topology:** `resolve_region_order`: played F4 games if present, else
  `seeds.f4_pairing`; never a default. Strict picks projection everywhere.
- **Opponents:** n = pool size − 1 (29 unless the pool's own history gives its size);
  per-team round pick shares (this pool's entries, else ESPN archive, else static
  seed rates); per-game P(pick t1) = share(t1)/(share(t1)+share(t2)); independent,
  path-consistent; redrawn every trial before any candidate is scored.
- **P(1st):** expected first-place share, ties split (1, 1/(1+k), 0), mean over
  shared CRN trials; = the winner-take-all prize. Denominator: trials (2,000 in the
  artifact; 500 selection trials / 100 evaluation repeats in the backtest).
- **Scoring:** ESPN 10/20/40/80/160/320, team identity, additive, no play-in points.
- **Production strategy `meta_region_poolaware`:** candidates = `region_top_n` ×
  bases {tv, mass_avg, mass_best, blend, tv_mass80} × risks {0.1, 0.3, 0.5, 0.7, 0.9}
  + `exhaustive_champion` × bases × {0.3, 0.5, 0.7}, byte-dedup first-label-kept;
  first argmax of P(1st) on 500 CRN trials (seed 77777+year). Shipped win-maximiser
  = `blend_region_35`.

## 2. Validation methodology (how 2027 will be evaluated, fixed now)

- Referee set = the Step 8 frozen qualification
  (`artifacts/methodology_audit/step8/qualification_v2.json`): {seed, torvik, blend,
  pit, market_v2}; independent referee **market_v2**; rule G1/G2 unchanged;
  incumbent = walk-forward seed.
- Criteria C1–C3 of `artifacts/referee_audit/PREREGISTRATION.md` unchanged;
  season-level paired bootstrap (5,000, seed 42); one fixed production bracket per
  season; LORO by the registered procedure.
- 2027 real-pool placement is reported separately from simulated P(1st) and is one
  observation; the A/B pre-registration `configs/frozen/prospective_2027_ab.json`
  (verified unchanged) governs the fixed-rule vs search comparison.
- Not permitted after 2027 outcomes are known: changing the referee set, the
  objective, the tie rule, the candidate families, the opponent model, or the gates;
  any such change is a new, dated pre-registration and applies only prospectively.

## 3. Future research (explicitly NOT in production)

R-1 heuristic noseed marginals; R-2 candidate-space expansion (+1.2 pp
out-of-sample measured); R-3 opponent sampler joint structure / real-entry
resampling (deep-round shares off by up to 12 pp); R-4 CLOSED negative
(recency/upset weights fail on both populations under frozen gates); R-5
production calibration slope 1.33 on 2023–25 (observation). Regular-season
games remain available as training data only if a separately pre-registered
experiment demonstrates incremental tournament value; the current evidence
rejects it.

## 4. 2027 data contract

- `data/raw/historical/tournament_context_2027.json`: the `seeds` block with
  canonical ids, regions, and **`f4_pairing` = [[A,B],[C,D]]** from the announced
  bracket (required; the builders raise without it); First Four results appended as
  `round_name = "FF"` rows before the Round of 64 (they finish before brackets lock).
- Selection Sunday snapshot = Torvik `data_type: pre_tournament` with
  `cutoff_date` before the first R64 tip; ESPN pick archive captured before tip
  (`require_archived`); Kaggle seeds file for 2027; no post-Selection-Sunday
  information anywhere in prediction inputs.
- Team ids: the existing canonical resolution (`normalize_team_id`, curated
  `KAGGLE_TEAMNAME_ALIASES`, market `ODDS_ID_ALIASES`); no fuzzy matching in any
  production path.

## 5. The conclusion, stated narrowly

The production strategy demonstrated robust historical performance under the
frozen validation framework, including against the materially different
qualified market referee (+4.0 pp [+2.5, +5.6] over the seed bracket, 13 of 14
seasons). This is historical validation. It is not evidence that the strategy
will achieve the same advantage in 2027 or in any real pool, and the referee
family provides one independent confirmation, not five.

## 6. Do not optimize during the tournament

Once the 2027 production snapshot is built from this specification, no change to
model selection, weighting, candidate space, referee, objective, scoring or
opponent model may be applied to 2027 production. Any such change requires a
separately dated pre-registration and applies only to later seasons.

## 7. Closeout status (2026-09-16) — addendum, spec unchanged

Recorded at retirement of the methodology audit. Nothing in sections 1-6 is
altered by this section.

- Production methodology: audited and corrected.
- Probability mathematics: audited.
- Tournament topology: corrected and propagated.
- Candidate space: audited; limitation quantified, not silently optimized.
- Opponent/ownership model: audited; known modelling limitation documented.
- Referee construction: audited; defective market referee retired, qualified set frozen.
- Referee robustness: independently supported under market_v2, with LORO.
- Recent-3-season / regular-season / upset hypothesis: pre-registered and rejected.
- Tournament-population recency/upset weighting: tested and rejected.
- Fitted-model UI comparison: uses the common evaluation framework
  (`scripts/evaluate_fitted_bracket.py`, exact parity against the shipped
  artifact, browser picks re-derived by executing the browser's own code)
  without entering production selection. Displayed EV and P(1st) reproduce the
  existing production definitions, which use different probability tables; a
  documented characteristic, not a unified model.
- Artifacts: rebuilt and parity-checked (14 seasons, exact).
- Fresh-checkout reproducibility: verified.
- Regression: green aside from the deliberately classified non-production CLI
  `optimize-pool --mode seed` case (`xfail`, strict).
- Remaining CLI/research defects: explicitly isolated from production and
  documented (`tests/test_topology_isolation.py`, `AGENT_NOTES.md`,
  `docs/SITE_REVIEW_TODO.md`).

**What PASS means.** Not that every file in the repository is defect-free. It
means the audited production and validation methodology is sound enough to
freeze, and that the known dead/research-path defects are proven -- by
dependency graph and by contract tests, not by the UI looking right -- to lie
outside the production dependency graph.

**Frozen.** No R-5. No new referee. No candidate-space expansion. No
opponent-sampler redesign. No new weighting experiment. No further
retrospective modelling tests. The next meaningful experiment is 2027
prospective performance under this specification: a clean out-of-sample test
in which a poor result cannot be attributed to an unresolved topology bug, a
stale artifact, leakage, an inconsistent scorer, a referee construction
defect, or a moving validation target.
