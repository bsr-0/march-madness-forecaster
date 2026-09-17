/* Bracket Lab — one page, one question: what are you optimising for.
 *
 *   Maximise chance of winning   a FIXED RULE, not a search: region-by-region
 *                                construction over a seed/no-seed probability
 *                                blend at a constant contrarian risk of 0.35.
 *                                Backtested at pool 30 across 2011-2026.
 *
 *   Maximise expected points     the exact expected-points maximum, solved by
 *                                dynamic programming on the bracket. Equivalent
 *                                to sending whichever team is likelier to win
 *                                the tournament through every game.
 *
 *   Fitted model                 a ridge SPREAD regression fitted live on
 *                                tournament games. Predicts scoring MARGIN in
 *                                points, with P(win) following as
 *                                Phi(margin / sigma); see fit.js. It is not a
 *                                classifier and a coefficient is not a log-odds.
 *
 * THE FIRST TWO ARE NOT THE SAME BRACKET AND THE DIFFERENCE IS THE POINT.
 * On the current 2026 artifact both name Michigan, but the win-maximiser
 * gives up about 40 expected points (902 against 943) to nearly double its
 * chance of finishing first (7% against 4%) by taking upsets the field will
 * not. (This comment used to cite 872/941 and 9.9%/3.9% with different
 * champions -- figures from an earlier artifact; the numbers on the page come
 * from the payload, never from here.) In a winner-take-all pool the points
 * number is worth nothing and the trade is free; in a pool paying second and
 * third it is a real decision. Both scores are shown for whichever strategy is
 * selected, so the cost is visible rather than implied.
 *
 * WHY THE FIRST IS A FIXED RULE. It used to be the best of ~3,000 candidates
 * scored by a P(1st) referee. That route had never been backtested, and its
 * headline number was the maximum of a noisy estimate and so biased upward. The
 * fixed rule is the one with out-of-sample evidence: at pool 30 it reaches
 * P(1st) ~0.10-0.11 at any risk in 0.2-0.5, against 0.064 for the same
 * construction on Torvik ratings and 0.040 for a seed bracket. Choosing the risk
 * level per season measured WORSE than fixing it, so 0.35 is the middle of a
 * plateau rather than an optimum.
 *
 * WHAT USED TO BE HERE, AND WHY IT IS NOT. The page let the user pick the
 * variables, then the model family, then the training matrix. All three are gone
 * and all three went for the same reason: measurement said the choosing bought
 * nothing, or bought something worse.
 *
 *   variables      per-fold selection scored 0.46651 against 0.45698 for the
 *                  fixed canonical set, inside the bootstrap's noise
 *   model family   ridge beat kNN k=25 (CI [-0.040, -0.011]), LightGBM
 *                  (CI [-0.018, -0.001]) and local linear outright; kNN at
 *                  k=100 and k=500 could not be separated from it. Nothing beat
 *                  ridge, so the control could only select something worse
 *   training set   pooling 41,321 regular-season rows measured null against the
 *                  1,008 tournament rows on the same walk-forward split
 *   history prior  blending toward the seed-matchup base rate was MONOTONICALLY
 *                  worse: 0.45454 at weight 0, 0.45566 at 0.1, 0.48189 at 0.5,
 *                  0.56138 at 1.0. No round benefited -- the two that looked
 *                  like they did, R32 (+0.0032) and E8 (+0.0055), were the best
 *                  of 21 weights on 189 and 41 games and neither survived a
 *                  bootstrap
 *
 * The prior was not noise: alone it scores 0.561 against a coin flip's 0.693.
 * It is simply a cruder measurement of what barthag and t_rank already carry,
 * so blending it in diluted rather than complemented. Worth remembering before
 * anyone adds a second source of seed information.
 *
 * Choosing an OBJECTIVE is a decision the data cannot make for you, and those
 * controls stayed. Choosing an ESTIMATOR is a decision it can, and those went.
 *
 * The fit excludes the displayed season and every later one (walk-forward,
 * not leave-one-year-out -- see fitLinear()'s docstring in fit.js), so the
 * coefficients were never derived from the games being predicted, or from
 * tournaments that had not been played yet.
 */

const ROUNDS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];
const MOBILE_ROUND_DEFAULT = ROUNDS.indexOf('Final Four');

/* The browser-fitted strategy. Anything else is a precomputed bracket read out
 * of the season payload by id. */
const MODEL = 'model';

/* The experimental rule-search strategy. Found after the fact, never scored
 * by the pool referee, fenced on the page; see ruleSearch() in fit.js and
 * the RULE SEARCH section below. */
const RULE = 'rule';
const RULE_CHECKPOINTS = [
  { r: 0, label: 'Round of 32' }, { r: 1, label: 'Sweet 16' }, { r: 2, label: 'Elite Eight' },
  { r: 3, label: 'Final Four' }, { r: 4, label: 'Finalists' }, { r: 5, label: 'Champion' },
];
/* Rounds that count as an early checkpoint: at least one must stay checked
 * (see setRuleCheckpoint). Finalists and champion alone leave rounds 0-3
 * unconstrained, keys^4 sequences per season before anything prunes. */
const RULE_EARLY_ROUNDS = [0, 1, 2, 3];
const RULE_N_MAX = 20;

/* Variables the fitted strategy uses.
 *
 * FIXED, NOT CHOSEN. This is the key set the frozen baseline in
 * artifacts/model_baseline.json is defined over, and it is what the shipped
 * accuracy number (log loss 0.45391 on held-out tournament games; 0.45296 before the 2026-09 correction of one 2025 results row) describes.
 * The UI used to let each variable be switched on and off, which meant the
 * board could be filled by a model no one had ever validated -- and measurement
 * said the choosing bought nothing: selecting features per fold scored 0.46651
 * against 0.45698 for this fixed set, a difference the bootstrap could not
 * separate from zero. Removing the control removes a decision that felt
 * meaningful and was not. */
const CANONICAL_KEYS = [
  'barthag', 't_rank', 'sos_avg_opp_barthag', 'adj_offensive_efficiency',
  'adj_defensive_efficiency', 'adj_tempo', 'effective_fg_pct', 'three_pt_pct',
  'three_pt_rate', 'offensive_reb_rate', 'turnover_rate',
];

const state = {
  // Replaced at init() by the newest season with status "ready". This literal
  // is only the pre-load placeholder: it used to be the actual default, which
  // meant that on Selection Sunday 2027 the page would open on the 2026
  // bracket with 2027 greyed out beside it -- a launch-day failure on the one
  // season the system was frozen to be judged on.
  year: 2026,
  strategy: 'p1',       // 'p1' | 'ev' | MODEL | CUSTOM
  /* CUSTOM is driven by this pair rather than by an id. Champion and depth are
   * independent properties of a bracket, so they filter JOINTLY: either alone
   * narrows the pool and both together narrow it further. They were previously
   * mutually exclusive menu entries, which made "Connecticut wins AND my Final
   * Four stops at a 3 seed" unaskable even though the pool carries 63 such
   * pairs. */
  pick: { champ: null, ones: null, depth: null, pred: null, src: null },
  /* Which of the matching brackets to show. The referee's standard error is
   * about half a point, so within a filtered set the top few are statistically
   * tied and picking only the argmax presents a coin flip as a verdict. */
  alt: 0,
  /* Which question the filters narrow. Filtering changes WHICH brackets are
   * eligible, never what is being maximised over them -- picking a champion
   * used to silently switch the objective to P(1st), which for Michigan meant
   * handing back a bracket worth 71 fewer expected points than the one the
   * user had asked for. */
  objective: 'p1',      // 'p1' | 'ev'
  fit: null,            // {beta, n, converged}
  advancement: null,    // {team: [P(reach R32), ..., P(win it all)]}, see refit()
  // Which round the narrow-viewport board is showing (see the @media rule in
  // app.css). Meaningless on a wide viewport, where CSS ignores it and every
  // round is visible regardless -- so there is nothing to gate on screen
  // width here, only to reset when the board underneath it changes shape.
  // Opens on the Final Four: on a phone the Round of 64 is 32 games of
  // scrolling before anything a visitor came for, and the champion, the
  // finalists and the semifinals are the end of the story, not the start.
  mobileRound: MOBILE_ROUND_DEFAULT,
  explore: 'adj_defensive_efficiency',   // variable key the Explore panel is showing
  // Rule search (experimental strategy): what the user asked a rule to
  // reproduce, over how many prior played seasons, and the last result.
  rule: {
    mode: 'search',                // 'search' | 'hand'
    checkpoints: [3, 4, 5],        // rounds whose winners a rule must reproduce
    from: null, to: null,          // fit range (played seasons strictly before the displayed one); null = last 3
    n: 5,                          // brackets to offer
    keys: null,                    // eligible criteria; null = every variable + seed
    rank: 'simple',                // 'simple' | 'outside' (most seasons outside the range reproduced)
    maxCriteria: null,             // offer only rules using at most this many distinct criteria; null = any
    hand: null,                    // by-hand rule: one criterion per round, 6 entries
    chosen: 0, result: null, busy: false,
    token: 0,                      // the latest ensureRuleSearch() call; earlier ones abandon when superseded
  },
  seasonsIndex: null,   // seasons.json, kept so the rule search knows which prior seasons were played
  training: null,
  season: null,
  priors: null,        // historical seed-matchup upset rates, per season
  cache: {},
};

/* ---------- data ---------- */

/* Cache key for EVERY file under data/.
 *
 * ONE CONSTANT, NOT ONE PER FILE. These payloads are regenerated together by
 * scripts/build_ui_payload.py, so per-file versions only create opportunities to
 * bump four of them and miss the fifth -- which has now happened three times in
 * this codebase: the priors file, app.js itself, and season_*.json when the
 * win-maximising strategy changed. The failure is silent every time. The deploy
 * succeeds, the new file sits on the server, and returning browsers keep reading
 * the old one, so the bug looks like "the site did not update" rather than an
 * error.
 *
 * BUMP THIS WHENEVER ANYTHING UNDER docs/data/ CHANGES. Over-bumping costs one
 * refetch of a few hundred KB; under-bumping ships wrong numbers to anyone who
 * visited before. */
const DATA_V = 21;

async function loadTraining() {
  if (state.training) return state.training;
  const res = await fetch(`data/training.json?v=${DATA_V}`);
  state.training = await res.json();
  return state.training;
}

async function loadSeason(year) {
  if (state.cache[year]) return state.cache[year];
  const res = await fetch(`data/season_${year}.json?v=${DATA_V}`);
  if (!res.ok) throw new Error(`season ${year} unavailable`);
  const data = await res.json();
  state.cache[year] = data;
  return data;
}

/* ---------- bracket solving ---------- */

/* Refit whenever the enabled set or the season changes.
 *
 * The displayed season is excluded from the fit. Without that the coefficients
 * would be derived from the very games being predicted, and the bracket would
 * look far better than the method deserves. */
function refit() {
  // ONE MATRIX, ONE MODEL, BOTH FIXED BY MEASUREMENT rather than offered as
  // choices, and the challengers were each given their best form before being
  // rejected. On this exact matrix and split:
  //
  //   ridge, 11 canonical keys                     0.45698
  //   LightGBM, best of n_estimators 20..800       0.51601   CI [-0.084, -0.033]
  //   kNN, best of 3 feature sets x 5 k values     0.53057   CI [-0.100, -0.047]
  //
  // THE FIRST VERSION OF THIS COMPARISON WAS UNFAIR AND ITS CONCLUSION STILL
  // HELD. It handed all 27 features to every model, which is close to neutral
  // for ridge (regularised) and for LightGBM (splits select implicitly) but
  // punishing for kNN, whose neighbourhoods dilute in high dimensions. Retested
  // properly, kNN does improve as features are cut -- 0.53458 at 11 features to
  // 0.53057 at 3 -- and that is worth 0.004 against a 0.074 deficit. LightGBM's
  // curve is flat from 120 trees to 800 (0.516 to 0.517), so its whole tuning
  // range is 0.016 while it trails by 0.059.
  //
  // Nothing beat ridge, so there was no choice to offer -- only a way to pick
  // something worse. The likely reason is the sample: 1,008 games with ~10.3
  // points of irreducible residual is a regime where eleven regularised
  // coefficients are about the right amount of structure, and extra flexibility
  // is spent on noise.
  //
  // Pooling regular-season rows measured null on the same split, so the
  // tournament matrix stands alone and training_pit.json (9 MB) is never
  // fetched.
  const wanted = CANONICAL_KEYS;
  const src = state.training;
  if (!src || !wanted.length) { state.fit = null; state.advancement = null; return; }

  // Variables the matrix cannot supply are dropped, not zero-filled: a zero
  // differential is a claim that the two teams are equal on it.
  const cols = [];
  const keys = [];
  for (const k of wanted) {
    const i = src.keys.indexOf(k);
    if (i >= 0) { keys.push(k); cols.push(i); }
  }
  if (!keys.length) { state.fit = null; state.advancement = null; return; }

  const f = fitLinear(src.games, cols, state.year);
  f.keys = keys;
  f.cols = cols;
  f.userKeys = keys;
  f.dropped = wanted.filter(k => src.keys.indexOf(k) < 0);   // e.g. t_rank has no dated snapshot
  // The honest number: fit on prior seasons, scored on seasons never seen --
  // and CALIBRATED on seasons strictly before the one on screen. Until the
  // 2026-09 audit this called crossValidate() on the whole matrix, so the
  // link's (a, nu) for a displayed 2019 had been fitted on 2019's own results
  // and on 2020-2026's. See causalWalkForward() in fit.js.
  f.oos = causalWalkForward(state.training.games, cols, state.training.years, state.year, 2014);
  // In-sample accuracy on the SAME games the walk-forward folds held out,
  // so the two numbers in the note are comparable. It used to show
  // fitQuality() -- every training row, 2010 onward -- while the folds start
  // at 2014. Set against
  // each other, that read as "held-out 78% beats in-sample 77.7%" -- true of
  // the numbers, meaningless as a comparison, and an invitation to read a
  // year-range artefact as evidence about overfitting (2026-09 site review).
  f.qualityOnFoldYears = null;
  if (f.oos) {
    const foldYears = new Set(Object.keys(f.oos.perYear).map(Number));
    const same = state.training.games.filter(r => foldYears.has(r.y));
    f.qualityOnFoldYears = scoreSpread(same, f.beta, cols);
  }
  // Overall rating and National rank, on the canonical set, correlate at
  // ~0.99 and draw large opposite-sign coefficients every season -- neither
  // ever flips sign across a fold, so stability()'s sign-flip check cannot
  // see it, and until this it went to screen unmarked (2026-09 site review).
  // See pairwiseCorrelations() in fit.js for why this is a different check.
  f.corr = pairwiseCorrelations(src.games, cols, state.year);
  state.fit = f;
  state.sens = null;   // exclusion refits are per fit; recomputed on demand by sensitivity()

  // Every team's chance of reaching every round, over the REAL bracket -- not
  // a seed-based base rate (that is a different question, answered on the
  // Python side for a different purpose). Computed once here, from the same
  // calibrated pairwise winProb() the board already grades games with, so the
  // per-round numbers in the team drawer can never disagree with the per-game
  // percentages on the board.
  //
  // refit() runs for every season regardless of status -- setYear() calls it
  // before render() has had a chance to bail out on a season that has not
  // started -- and a `not_started` season's payload carries no `first_round`
  // at all (see docs/data/season_2027.json before Selection Sunday). Without
  // this check that reached bracketAdvancementProbs() as `undefined.length`.
  const hasBracket = state.season && Array.isArray(state.season.first_round);
  state.advancement = fitReady() && hasBracket
    ? bracketAdvancementProbs(state.season.first_round, winProb) : null;
}

/* Predicted scoring margin for team a against team b, in points.
 *
 * Antisymmetric by construction: swapping a and b negates the differential and
 * so negates the margin exactly. */
/* The matchup's standardised differential on the enabled variables, in the
 * order fit.keys lists them. The ridge model dots this with beta; kNN uses it
 * as a query point. Both need the same vector, so it is built once here. */
function diffVector(a, b) {
  const z = state.season.z, f = state.fit;
  return f.keys.map(k => {
    // Venue is zero on a neutral court, which every NCAA game is. This is the
    // prediction-time counterpart of tournament_venue() on the Python side.

    const col = z[k];
    const d = col ? (col[a] || 0) - (col[b] || 0) : 0;
    return d;
  });
}

function margin(a, b) {
  const f = state.fit;
  const x = diffVector(a, b);
  let t = 0;
  for (let j = 0; j < f.keys.length; j++) t += f.beta[j] * x[j];
  return t;
}

/* P(team a beats team b): the predicted margin read against the fit's own
 * residual spread. A 6-point edge is near-certain for a model that is usually
 * within 2 points and a coin flip for one that is usually within 12, so the
 * spread is what carries the margin into a probability.
 *
 * The spread alone was not enough. The link is calibrated on held-out games --
 * a fitted scale and tail weight, see calibrate() in fit.js -- because the raw
 * in-sample sigma left the model measurably under-confident from 0.6 to 0.9 and
 * pinned its most lopsided picks against 1.0. Passing the calibration here is
 * what makes the board's percentages mean what they say. */
function winProb(a, b) {
  const cal = state.fit.oos && state.fit.oos.calibration;
  return winProbFromMargin(margin(a, b), state.fit.sigma, cal);
}

/* Play the bracket out under the fit. Exact ties go to the better seed, then
 * lower index, so the board never jitters on a coin-flip game. */
function solveByFit() {
  return solveBracket(winProb);
}

/* The same walk with any P(a beats b). Exists so the sensitivity panel can
 * re-solve under an exclusion model with the SAME tie rule as the board,
 * rather than a second, slightly different walk. */
function solveBracket(pFn) {
  const teams = state.season.teams;
  let current = state.season.first_round.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const p = pFn(a, b);
      let win;
      if (p !== 0.5) win = p > 0.5 ? a : b;
      else if (teams[a].seed !== teams[b].seed) win = teams[a].seed < teams[b].seed ? a : b;
      else win = Math.min(a, b);
      games.push({ a, b, win, p });
      next.push(win);
    }
    rounds.push(games);
    current = next;
  }
  return rounds;
}

/* Expand a precomputed strategy's picks into the same shape as the fitted board. */
function solveFromPicks() {
  const s = currentStrategy();
  const src = s ? s.picks : state.season.pool_optimized;
  const picks = src.map(r => new Set(r));
  let current = state.season.first_round.slice();
  const rounds = [];
  // Computed once per solve, not per game: fit.js is not re-fitted here, only
  // queried, so this is a flag read, not a cost.
  const fitOk = fitReady();
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const ha = picks[r].has(a), hb = picks[r].has(b);
      // Exactly one of the two teams must be this round's winner. The old
      // `has(a) ? a : b` silently invented a winner whenever the picks did not
      // describe a bracket on this tree (2026-09 audit, Step 4, F4-7).
      if (ha === hb) {
        throw new Error(`picks do not describe a bracket on this season's tree: round ${r}, game ${a} vs ${b}, ${ha ? 'both' : 'neither'} picked`);
      }
      const win = ha ? a : b;
      // This game's own probability is not in the payload -- the precomputed
      // strategies carry a whole-bracket P(1st)/EV, not a per-game figure. The
      // live fitted model can score any pair, so it supplies the confidence
      // number here too; see fitReady() for why that is the right source
      // rather than leaving these boards silent.
      const p = fitOk ? winProb(a, b) : null;
      games.push({ a, b, win, sa: null, sb: null, p });
      next.push(win);
    }
    rounds.push(games);
    current = next;
  }
  return rounds;
}

/* Play out what actually happened, in the same shape as the model's bracket.
 *
 * Needed because "was this pick right" is a question about a SLOT, not just a
 * team: Duke reaching the Elite 8 in reality does not make the model right if
 * the model had Duke in a different half of the draw. Solving reality on the
 * same structure lets every game be compared position by position.
 */
function solveActual() {
  const a = state.season.actual;
  if (!a) return null;
  const won = a.map(r => new Set(r));
  let current = state.season.first_round.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const x = current[g], y = current[g + 1];
      games.push({ a: x, b: y });
      // A slot is only real while reality is still following this path.
      next.push(won[r].has(x) ? x : won[r].has(y) ? y : null);
    }
    rounds.push(games);
    current = next;
  }
  return rounds;
}

/* Strategies are alternatives, not layers: a precomputed bracket comes from the
 * validated selector, the fitted one is built here. Exactly one is on screen. */
/* The active bracket, whichever kind it is.
 *
 * Champion picks are resolved here rather than being copied into `strategies`
 * so there is exactly one list of them, and so the note and the board cannot
 * disagree about which bracket is showing. */
const CUSTOM = 'custom';

/* Filtering happens here rather than in a precomputed table.
 *
 * The payload used to ship a cell per subset of the filter axes, which grew
 * multiplicatively: three axes were 282 cells, four 675, five 1,496. It now
 * ships the candidates themselves with their attributes, so a filter is a scan
 * and a new axis is one more field. Selecting a maximum over a filtered array
 * is presentation arithmetic; the modelling that produced the candidates stays
 * in Python.
 */
const AXIS_FIELD = { champ: 'c', ones: 'o', depth: 'd', src: 's' };

/* Preference predicates are a bit string rather than a scalar, so they are
 * matched separately. These come from src/product/selection.py -- the same
 * definitions the artifact scores its constraint coverage against -- rather
 * than being recomputed here, so a predicate added there reaches the page
 * without a second implementation drifting away from it. */
function matchesPred(row, i) {
  return i === null || row.k[i] === '1';
}

function candidates() {
  return ((state.season && state.season.filters) || {}).candidates || [];
}

function matching(pick) {
  const p = pick || state.pick;
  return candidates().filter(r =>
    Object.entries(AXIS_FIELD).every(([k, f]) => p[k] === null || r[f] === p[k])
    && matchesPred(r, p.pred));
}

function anyFilter() {
  // `pred` is matched separately from AXIS_FIELD because it is a bit string
  // rather than a scalar, and it has to be counted here too. It was not, so
  // selecting a bracket shape on its own set the chip active while leaving the
  // strategy unfiltered -- the board kept showing the recommended bracket and
  // the control looked broken in the one way that is hard to notice: it did
  // nothing.
  return Object.keys(AXIS_FIELD).some(k => state.pick[k] !== null) || state.pick.pred !== null;
}

/* A bracket is 63 binary choices against the known first-round order. */
function decodeBracket(bits) {
  const fr = state.season.first_round;
  const rounds = [];
  let cur = fr.slice(), i = 0;
  for (let r = 0; r < 6; r++) {
    const nxt = [];
    for (let g = 0; g < cur.length; g += 2) {
      const t1 = cur[g], t2 = cur[g + 1];
      const w = bits[i++] === '1' ? t1 : t2;
      nxt.push(w);
    }
    rounds.push(nxt);
    cur = nxt;
  }
  return rounds;
}

const SRC_LABEL = {
  torvik: 'Torvik', massey_avg: 'Massey', elo: 'Elo',
  region_top_n: 'region construction', shipped: 'the recommended brackets',
};

function filteredEntry() {
  // WITH NOTHING SELECTED THERE IS NO FILTERED SET. matching() returns every
  // candidate in that case, which is correct as a query and wrong as an answer:
  // it made both strategy cards show the pool-wide best and label it
  // "filtered", so they read as identical and as narrowed when neither was
  // true. An empty filter is not a filter.
  if (!anyFilter()) return { entry: null, scope: '', alts: [] };
  const rows = matching();
  if (!rows.length) return { entry: null, scope: '', alts: [] };
  const obj = state.objective;
  // NEAR-TIED IS DEFINED BY THE REFEREE'S ERROR, NOT BY A FIXED COUNT. The
  // first version showed a top-3 and called them tied; for 2026's depth=3 the
  // 1st and 3rd were 1.7 SE apart, so that label was doing work the numbers did
  // not support. Only candidates within one standard error of the best are
  // offered, capped at 5, so a genuine gap collapses the list to one entry
  // rather than dressing a ranking up as a choice.
  const sorted = rows.slice().sort((a, b) => b[obj] - a[obj]);
  const se = obj === 'p1' ? p1StandardError(sorted[0].p1) : 0;
  // Two estimates, not one: "is B distinguishable from the best?" needs the
  // error on the DIFFERENCE. sqrt(2) is that factor for independent samples and
  // is therefore conservative here, because candidates are scored on common
  // random numbers (FINDINGS 6e) which cancels part of the noise -- the one
  // place it was measured, the difference SE was ~1.2x rather than 1.41x the
  // single-estimate SE. Conservative is the right direction for this feature:
  // it exists so the argmax is not presented as a verdict when it is a coin
  // flip. On 2026 all three multipliers select the same four brackets.
  const cut = sorted[0][obj] - Math.SQRT2 * se;
  const alts = se > 0
    ? sorted.filter(r => r[obj] >= cut).slice(0, 5)
    : sorted.slice(0, 3);
  const pick = alts[Math.min(state.alt, alts.length - 1)];
  const { champ, ones, depth, pred, src } = state.pick;
  const bits = [];
  if (champ !== null) bits.push(`${state.season.teams[champ].name} winning`);
  if (ones !== null) bits.push(`${ones} one-seed${ones === 1 ? '' : 's'} in the Final Four`);
  if (depth !== null) bits.push(`a Final Four reaching exactly a ${depth} seed`);
  if (pred !== null) {
    const pd = (state.season.filters.predicates || []).find(x => x.i === pred);
    if (pd) bits.push(pd.label.toLowerCase());
  }
  if (src !== null) bits.push(`brackets from ${SRC_LABEL[src] || src}`);
  return {
    entry: {
      n: rows.length,
      row: pick,
      // The cards ask "what would I get if I asked THIS question of the set I
      // have narrowed to", so each objective needs its own best -- not the
      // current objective's pick echoed twice, which made both cards show the
      // same number and hid the trade the two strategies exist to express.
      by: {
        p1: rows.reduce((a, b) => (b.p1 > a.p1 ? b : a)),
        ev: rows.reduce((a, b) => (b.ev > a.ev ? b : a)),
      },
    },
    scope: bits.join(' and '),
    alts,
  };
}

function currentStrategy() {
  if (state.strategy === CUSTOM) {
    const { entry, scope, alts } = filteredEntry();
    if (!entry) return null;
    const obj = state.objective;
    const src = entry.row;
    const objName = obj === 'ev' ? 'expected points' : 'P(1st)';
    return {
      id: CUSTOM,
      label: obj === 'ev' ? 'Most points, your filters' : 'Best odds, your filters',
      // The qualifying count is shown deliberately. As filters narrow, the best
      // survivor is chosen from fewer candidates, and best-of-11 sits closer to
      // the maximum of a short noisy sample than to an optimum.
      note: `The highest-${objName} bracket with ${scope}, out of ${entry.n} qualifying candidates.`
          + (alts.length > 1
              ? ` Showing ${Math.min(state.alt, alts.length - 1) + 1} of ${alts.length} that score too close together to separate.`
              : ''),
      picks: decodeBracket(src.b), ev: src.ev, p1: src.p1,
      alts: alts.length, altIndex: Math.min(state.alt, alts.length - 1),
      source: SRC_LABEL[src.s] || src.s,
    };
  }
  if (state.strategy === RULE) return ruleStrategy();
  const list = (state.season && state.season.strategies) || [];
  return list.find(s => s.id === state.strategy) || null;
}

/* The standard error on a P(1st) estimate AT THAT ESTIMATE'S OWN p.
 *
 * The payload also carries `p1_se`, a scalar the artifact computes once at a
 * fixed reference of p=0.05. Applied to a candidate at p=0.099 that understates
 * the error by 37% (0.49pp against 0.67pp), and it is the number that decides
 * which brackets are offered as near-tied. Binomial SE from the trial count is
 * presentation arithmetic on a value already in the payload; no model math
 * moves into the browser.
 */
function p1StandardError(p) {
  const f = (state.season && state.season.filters) || {};
  const n = f.p1_trials;
  if (!n || !(p > 0) || !(p < 1)) return f.p1_se || 0;
  return Math.sqrt((p * (1 - p)) / n);
}

/* P(1st), printed no finer than it is known.
 *
 * toFixed(1) implies a resolution of 0.05pp against a standard error of about
 * 0.7pp -- roughly fourteen times finer than the number's own error, on every
 * chip and card. Whole points are still enough to separate the strategies
 * (10% against 4%) without inviting a user to read 9.9 as beating 9.8.
 */
function p1Pct(p) {
  // Whole points, except at the bottom: rounding 0.4% to "0%" reads as
  // impossible rather than unlikely, and three of 2026's candidates land there.
  if (p > 0 && p * 100 < 0.5) return '<1%';
  return `${(p * 100).toFixed(0)}%`;
}

/* Which season to open on.
 *
 * Deliberately not max(year): 2027 is listed as soon as the calendar knows
 * about it and stays "not_started" until Selection Sunday, so the newest LISTED
 * season is an empty state for most of the year. And deliberately not a
 * literal: a hardcoded 2026 would have opened the 2027 tournament on last
 * year's bracket.
 *
 * Split out of init() so it can be tested without a network.
 */
function pickDefaultSeason(seasons) {
  const ready = (seasons || []).filter(s => s.status === 'ready').map(s => s.year);
  if (ready.length) return Math.max(...ready);
  // No bracket anywhere: show the newest thing we know about and let its own
  // empty state explain itself, rather than a year that may not be listed.
  const all = (seasons || []).map(s => s.year);
  return all.length ? Math.max(...all) : null;
}

/* ---------- addressable state ----------
 *
 * A static site's only sharing surface is its URL. Until now the entire
 * selection lived in an in-memory object: reloading the page, or sending it to
 * the person running the pool, lost the bracket. The most common thing a user
 * wants to do with a bracket they like is show it to someone.
 *
 * Only presentation state is encoded -- which season, which question, which
 * filters, which alternate. Nothing here can change what the pool CONTAINS, so
 * a hand-edited hash cannot manufacture a bracket the model did not produce; at
 * worst it selects nothing and falls back.
 */
/* How many options in each row have nothing left behind them, so the panel can
 * say so in words. */
const _unavailable = {};

const HASH_KEYS = ['champ', 'ones', 'depth', 'pred', 'src'];

function writeHash() {
  if (!state.season) return;
  const p = new URLSearchParams();
  p.set('y', String(state.year));
  p.set('o', state.objective);
  if (state.strategy === MODEL) p.set('s', 'model');
  if (state.strategy === RULE) p.set('s', 'rule');
  for (const k of HASH_KEYS) {
    if (state.pick[k] !== null && state.pick[k] !== undefined) p.set(k, String(state.pick[k]));
  }
  if (state.alt) p.set('alt', String(state.alt));
  if (state.strategy === RULE) writeRuleHash(p);
  // replaceState, not a hash assignment: every chip click would otherwise add a
  // history entry, and Back would walk the user through their own filtering
  // one click at a time instead of leaving the page.
  history.replaceState(null, '', `#${p.toString()}`);
}

/* The rule search's inputs, in the URL with the rest of the selection. Like
 * the filters, these are presentation state: they choose what is searched
 * for and which of the offered brackets is shown, and nothing here can make
 * the search return a rule that does not reproduce what it was asked to.
 * Defaults are left out so an untouched panel adds nothing to the link. Keys
 * and hand criteria are restored as strings and checked against the season
 * by ruleKeys()/ruleHand(); a link naming a variable the season lacks falls
 * back rather than reaching rulePlay() with an unknown key. */
function writeRuleHash(p) {
  const r = state.rule;
  if (r.mode === 'hand') {
    p.set('rm', 'hand');
    if (r.hand) p.set('rh', r.hand.join(','));
  } else {
    if (r.from !== null && r.to !== null) { p.set('rf', String(r.from)); p.set('rt', String(r.to)); }
    if (r.n !== 5) p.set('rn', String(r.n));
    if (r.keys) p.set('rk', r.keys.join(','));
    if (r.rank === 'outside') p.set('rr', 'outside');
    if (r.maxCriteria !== null) p.set('rx', String(r.maxCriteria));
    if (r.chosen) p.set('ri', String(r.chosen));
  }
  p.set('rc', r.checkpoints.join(''));
}

function readRuleHash(p) {
  const r = state.rule;
  if (p.get('rm') === 'hand') r.mode = 'hand';
  const rc = p.get('rc');
  if (rc !== null) {
    // Same guard as setRuleCheckpoint: a link without an early checkpoint
    // keeps the default rather than launching an unprunable search.
    const cps = [...new Set(rc.split('').map(Number))].filter(x => RULE_CHECKPOINTS.some(c => c.r === x)).sort((a, b) => a - b);
    if (cps.some(x => RULE_EARLY_ROUNDS.includes(x))) r.checkpoints = cps;
  }
  const rf = parseInt(p.get('rf'), 10), rt = parseInt(p.get('rt'), 10);
  if (Number.isFinite(rf) && Number.isFinite(rt)) { r.from = rf; r.to = rt; }   // clamped to played seasons by ruleRange()
  const rn = parseInt(p.get('rn'), 10);
  if (Number.isFinite(rn)) r.n = Math.max(1, Math.min(RULE_N_MAX, rn));
  const rk = p.get('rk');
  if (rk) r.keys = rk.split(',').filter(Boolean).sort();
  if (p.get('rr') === 'outside') r.rank = 'outside';
  const rx = parseInt(p.get('rx'), 10);
  if (Number.isFinite(rx)) r.maxCriteria = Math.max(1, Math.min(ROUNDS.length, rx));
  const ri = parseInt(p.get('ri'), 10);
  if (Number.isFinite(ri) && ri >= 0) r.chosen = ri;                         // clamped to the offered list by ruleStrategy()
  const rh = p.get('rh');
  if (rh) {
    const hand = rh.split(',');
    if (hand.length === ROUNDS.length) r.hand = hand;
  }
}

/* Returns the season to open, or null to fall back to the newest ready one. */
function readHash() {
  const raw = (location.hash || '').replace(/^#/, '');
  if (!raw) return null;
  const p = new URLSearchParams(raw);
  const year = parseInt(p.get('y'), 10);
  if (p.get('o') === 'ev' || p.get('o') === 'p1') {
    state.objective = p.get('o');
    state.strategy = state.objective;
  }
  if (p.get('s') === 'model') state.strategy = MODEL;
  if (p.get('s') === 'rule') { state.strategy = RULE; readRuleHash(p); }
  for (const k of HASH_KEYS) {
    const v = p.get(k);
    if (v === null) continue;
    // Only `src` is a string ("torvik", "elo", ...). champ is a TEAM INDEX,
    // and ones/depth/pred are integers. Restoring champ as the string "9" would
    // fail every `===` against the payload's 9 and drop the filter in silence,
    // which is the failure a shared link is least likely to survive and least
    // likely to report.
    state.pick[k] = k === 'src' ? v : Number(v);
  }
  const alt = parseInt(p.get('alt'), 10);
  if (Number.isFinite(alt)) state.alt = alt;

  // Restoring the filters is not enough: setFilter also switches the page into
  // CUSTOM, and without that the link came back with the chips lit and the
  // board still showing the UNFILTERED bracket -- Florida selected, Michigan on
  // screen. A shared link that shows a different bracket than it promised is
  // worse than one that shows nothing.
  if (state.strategy !== MODEL && state.strategy !== RULE) state.strategy = anyFilter() ? CUSTOM : state.objective;

  // And show the controls that are evidently active, or the filtering looks
  // like the site's own opinion.
  if (anyFilter()) {
    const tune = document.getElementById('tune');
    if (tune) tune.open = true;
  }
  return Number.isFinite(year) ? year : null;
}

/* A shared link can outlive the thing it points at.
 *
 * The filters are indices into a season's candidate pool, so a link made before
 * an artifact rebuild -- or hand-edited, or moved to another season -- can name
 * a champion or a shape that no longer resolves. That used to leave the page in
 * CUSTOM with filters matching nothing, which rendered the ORDINARY default
 * bracket under the ordinary "chosen to maximise..." note: the visitor was told
 * they were looking at the shared bracket while looking at something else. It
 * is the same failure as restoring filters without CUSTOM, arriving from the
 * other side.
 *
 * Called once the season is loaded, because until then there is nothing to
 * resolve against.
 */
function reconcileFiltersWithSeason() {
  if (!state.season || state.season.status !== 'ready') return;
  if (!anyFilter() || state.strategy === MODEL || state.strategy === RULE) return;
  if (matching().length) return;

  state.pick = { champ: null, ones: null, depth: null, pred: null, src: null };
  state.alt = 0;
  state.strategy = state.objective;
  state.notice =
    'That link points at a bracket this season no longer has, so it is showing '
    + 'the standard recommendation instead.';
  writeHash();
}

function usingOptimized() {
  return state.strategy !== MODEL;
}

/* The strategies whose bracket comes out of the precomputed pool, which is
 * what the Adjust filters narrow: the two objectives and a filtered pick.
 * The fitted model solves its own bracket and the rule search applies a
 * rule, so the filters have nothing to act on under either. */
function usingFilters() {
  return state.strategy !== MODEL && state.strategy !== RULE;
}

function anyEnabled() {
  return !usingOptimized() && state.fit && state.fit.keys.length > 0;
}

/* Whether the live fitted model (fit.js) can score an arbitrary matchup right
 * now, regardless of which strategy is on screen.
 *
 * Distinct from anyEnabled(), which additionally requires the Fitted strategy
 * to be ACTIVE -- that gate is right for the equation and its prose, which are
 * claims about that specific model run, but wrong for game percentages: the
 * two precomputed strategies pick winners from a Python-side artifact that
 * carries no per-game probability, so the calibrated live model is the only
 * source of a confidence number for their games too. */
function fitReady() {
  return !!(state.fit && state.fit.ok && state.fit.keys && state.fit.keys.length > 0);
}

/* The fitted bracket's P(1st)/EV under the production referee -- IF, and only
 * if, it is the bracket this page just solved.
 *
 * scripts/evaluate_fitted_bracket.py scores the fitted bracket with the same
 * scorer, same referee tables, same 29-opponent pool and same trials the two
 * precomputed cards were scored with (it proves that by re-scoring those
 * cards' own brackets first and demanding exact equality). The payload
 * builder already refuses to embed an evaluation whose inputs have changed;
 * this is the last line: compare the 63 picks in the evaluation to the 63
 * picks solveByFit() produces right now, and show the numbers only on an
 * exact match. A P(1st) for a bracket that is not the one on screen is not
 * a slightly wrong number, it is a number about something else.
 *
 * WHAT THE NUMBER MEANS. It is the P(1st) of this bracket when evaluated in
 * the common pool framework the other cards are scored in -- not the fitted
 * model's own belief about its chances. The two would only coincide if the
 * pool referee were this model, and it is not (it is the seed-rate referee
 * with an ESPN-crowd opponent field). The copy on the card says so.
 *
 * Returns the payload's fitted_eval block, or null. `stale` is true when an
 * evaluation exists but is for a different bracket, so the UI can say that
 * rather than silently showing nothing. */
function fittedEval() {
  const s = state.season;
  const fe = s && s.fitted_eval;
  if (!fe || fe.kind !== 'fitted_model_evaluated' || !fitReady()) return null;
  const rounds = solveByFit();
  const same = fe.w.length === rounds.length && fe.w.every((r, i) =>
    r.length === rounds[i].length && r.every((t, j) => t === rounds[i][j].win));
  return same ? fe : { stale: true };
}

function fittedEvalNote() {
  const fe = fittedEval();
  if (!fe) return '';
  if (fe.stale) {
    return ` <span class="tag alt">Not scored</span> The pool evaluation on file is for a ` +
      `different bracket than this one, so no P(1st) or expected points are shown for it.`;
  }
  return ` <span class="tag alt">Evaluated, not selected</span> Scored in the same 30-entry pool ` +
    `framework as the two cards above: <strong>${p1Pct(fe.p1)}</strong> chance of finishing first, ` +
    `<strong>${fe.ev.toFixed(0)}</strong> expected points. That is the P(1st) of this bracket under ` +
    `the common referee, not the model's own belief about its chances \u2014 and it was scored ` +
    `there, not selected there: it is not one of the candidate brackets.`;
}

/* ---------- render ---------- */

function render() {
  const s = state.season;
  const board = document.getElementById('board');
  const empty = document.getElementById('empty');
  const note = document.getElementById('mode-note');
  const weights = document.getElementById('strategy');
  if (!s || s.status !== 'ready') {
    board.innerHTML = '';
    state.rounds = null;
    { const tools = document.getElementById('board-tools'); if (tools) tools.hidden = true; }
    { const nav = document.getElementById('board-nav'); if (nav) nav.hidden = true; }
    { const dots = document.getElementById('rnav-dots'); if (dots) dots.hidden = true; }
    { for (const id of ['headline', 'compare', 'why']) { const el = document.getElementById(id); if (el) { el.hidden = true; if (id !== 'why') el.innerHTML = ''; } } }
    { const ex = document.getElementById('explore'); if (ex) ex.hidden = true; }
    { const rp = document.getElementById('rulepanel'); if (rp) rp.hidden = true; }
    { const tune = document.getElementById('tune'); if (tune) tune.hidden = true; }
    weights.hidden = true;
    { for (const id of ['champions', 'ones', 'shapes', 'dd16', 'sources', 'alts']) {
        const el = document.getElementById(id); if (el) el.hidden = true; } }
    note.innerHTML = '';
    // Both of these belong to a bracket. Leaving them up under "the 2027 season
    // hasn't started yet" put a P(1st) disclosure and a stale "your filter was
    // dropped" message on the exact screen a visitor sees for the five months
    // before Selection Sunday.
    { const el = document.getElementById('p1-note'); if (el) el.textContent = ''; }
    { const el = document.getElementById('dropped-note'); if (el) el.textContent = ''; }
    state.notice = '';
    empty.hidden = false;
    empty.innerHTML = `
      <p class="e-title">${s ? s.message : 'Season unavailable.'}</p>
      <p class="e-sub">${s ? s.detail : ''}</p>`;
    return;
  }

  empty.hidden = true;
  weights.hidden = false;   // the panel is the only control surface
  // The filters narrow the precomputed pool. The fitted model and the rule
  // search do not draw from it, so under those two the Adjust panel is not
  // offered at all rather than shown with nothing to act on (2026-09-17;
  // it used to stay up, dimmed under the model and live under the rule
  // search, where a click silently switched strategy).
  { const tune = document.getElementById('tune'); if (tune) tune.hidden = !usingFilters(); }
  // The prior blend applies to the fitted board only. The Optimized picks are
  // precomputed and are not a regression, so there is nothing to blend into.


  if (usingOptimized()) {
    const st = currentStrategy();
    // Both scores, always, for whichever strategy is showing. A bracket built to
    // win outright gives up real expected points to do it, and stating only the
    // number its own objective optimises would hide exactly that cost.
    // Three different kinds of claim, and the label must not launder one as
    // another. The fixed rule has out-of-sample backtest evidence; the expected
    // points bracket is an exact solution; a champion pick is the best-scoring
    // member of the candidate pool for a belief the USER supplied, which is not
    // a validated recommendation at all.
    const kind = !st ? 'Tested on past seasons'
      : st.id === RULE ? 'Experimental'
      : st.id === CUSTOM ? 'Your pick'
      : st.id === 'ev' ? 'Exact optimum'
      : 'Backtested rule';
    note.innerHTML = `<span class="tag${st && st.id === RULE ? ' alt' : ''}">${kind}</span><span>${st ? st.note : s.pool_optimized_note}` +
      (st && st.id !== RULE ? ` <strong>${p1Pct(st.p1)}</strong> chance of finishing first, ` +
            `<strong>${st.ev.toFixed(0)}</strong> expected points.` : '') + `</span>`;
  } else if (!anyEnabled()) {
    note.innerHTML = `<span class="tag alt">Unavailable</span><span>The fitted model needs training data for seasons before ${state.year}.</span>`;
  } else {
    const f = state.fit, o = f.oos;
    // Lead with out-of-sample. In-sample is shown second and labelled, because
    // it always looks better and always will.
    note.innerHTML = `<span class="tag alt">Fitted</span><span>` +
      `${f.keys.length} variable${f.keys.length > 1 ? 's' : ''}, fitted on ${f.n.toLocaleString()} games from seasons before ${state.year}. ` +
      (o ? `Across ${o.seasons} held-out seasons it is off by <strong>${o.mae.toFixed(1)} points</strong> in a typical game ` +
           `and calls <strong>${(o.accuracy * 100).toFixed(0)}%</strong> of them correctly ` +
           // Same games both times, and whole points both times: with ~700
           // games the standard error on an accuracy is about 1.5pp, so a
           // decimal place implies a resolution the number does not have.
           (f.qualityOnFoldYears
             ? `— against ${(f.qualityOnFoldYears.accuracy * 100).toFixed(0)}% on those same games when they were in the training set.`
             : '.') +
           // Accuracy grades the pick; the board also shows a percentage, and
           // that is a separate claim needing a separate number.
           // Accuracy grades the pick; the percentage on the board is a
           // separate claim. This used to assert "when it says 70% it is right
           // about 70% of the time" with nothing on the page behind it (2026-09
           // review). The table under the equation now IS the evidence, so the
           // sentence points at it instead of vouching.
           (o.reliability ? ` Its percentages are a separate claim from its picks \u2014 ` +
             `the table under the equation checks them band by band on those same unseen games.` : '')
         : `Not enough history to test out-of-sample.`) +
      fittedEvalNote() +
      `</span>`;
  }

  // The disclosure rides with the numbers. Shown whenever a P(1st) figure is on
  // screen, which is every "% to win" on the strategy cards and every chip.
  const drop = document.getElementById('dropped-note');
  if (drop) drop.textContent = state.notice || '';

  const p1note = document.getElementById('p1-note');
  if (p1note) p1note.textContent = s.p1_assumption || '';

  document.getElementById('equation').innerHTML = anyEnabled() ? equationHTML() : '';

  if (state.strategy === RULE && !ruleStrategy()) {
    // Nothing to draw yet: the search is fetching prior seasons or found no
    // rule. The panel says which; the board stays empty rather than showing
    // some other strategy's bracket under this label.
    state.rounds = null;
    board.innerHTML = '';
    { const tools = document.getElementById('board-tools'); if (tools) tools.hidden = true; }
    { const el = document.getElementById('headline'); if (el) el.hidden = true; }
    // The strategy table stays: it is the way to another strategy, and
    // hiding it with the board used to strand the visitor here.
    renderCompare();
    updateMobileNav();
    renderRulePanel();
    return;
  }
  const rounds = usingOptimized() ? solveFromPicks() : solveByFit();
  const truth = solveActual();
  // Kept for the exporter. Presentation state only -- copyPicks() serialises
  // exactly what is on screen rather than re-deriving it, so the two can never
  // disagree about which bracket the user is looking at.
  state.rounds = rounds;
  { const tools = document.getElementById('board-tools'); if (tools) tools.hidden = false; }
  renderHeadline(rounds);
  renderCompare();
  renderExplore();
  renderRulePanel();
  { const why = document.getElementById('why'); if (why) why.hidden = false; }
  // `active`/`data-r` matter only under the narrow-viewport CSS (see
  // app.css's @media (max-width: 720px)), which shows one .round at a time
  // instead of scrolling six columns sideways. They cost nothing on a wide
  // viewport, where that rule never applies and every round is visible
  // regardless of this class.
  board.innerHTML = rounds.map((games, r) => `
    <div class="round${r === state.mobileRound ? ' active' : ''}" data-r="${r}" style="--n:${games.length}">
      <p class="r-label">${ROUNDS[r]}${ruleRoundLabel(r)}</p>
      ${games.map((g, gi) => gameHTML(g, r, truth ? truth[r][gi] : null)).join('')}
    </div>`).join('');
  updateMobileNav();
}

/* ---------- headline and comparison ----------
 *
 * The page used to open on three explanatory cards, a disclosure, the filter
 * stack, a strategy note, an equation and only then the bracket: a research
 * dashboard that happened to emit a bracket. The methodology audit
 * (artifacts/methodology_audit/) made the numbers on this page much stronger
 * and changed almost nothing a visitor could see, which was the right
 * outcome for an audit and the wrong state for a product. The structure is
 * now: what should I pick (headline) -> why this one and what else is there
 * (comparison table) -> the bracket -> adjust -> how it was tested
 * (collapsed). Nothing here computes anything new: every number is the same
 * payload field the strategy cards already show, read through strategyRows()
 * so the headline, the table, the cards and the board cannot disagree.
 */

/* One row per selectable bracket, from the same fields the cards use.
 *
 * With filters active the two precomputed rows show the FILTERED best for
 * their objective (the same `by` values the cards show, see filteredEntry()),
 * because the row a click returns is that filtered bracket. The fitted row's
 * champion is always the live fit's; its P(1st)/EV exist only when the
 * evaluation on file is for exactly that bracket (fittedEval()). */
function strategyRows() {
  const s = state.season;
  const teams = s.teams;
  const filt = state.strategy === MODEL ? null : filteredEntry().entry;
  const rows = (s.strategies || []).map(st => {
    const v = (filt && filt.by && filt.by[st.id]) || st;
    const champIdx = filt && filt.by && filt.by[st.id] ? decodeBracket(v.b)[5][0] : st.picks[5][0];
    const filtered = !!(filt && filt.by && filt.by[st.id]);
    return {
      id: st.id,
      label: st.id === 'ev' ? 'Most expected points' : 'Win the pool',
      kind: st.id === 'ev' ? 'Exact optimum' : 'Backtested rule',
      p1: v.p1, ev: v.ev, champion: teams[champIdx],
      filtered,
      // The realised result is for the STRATEGY's bracket; a filtered row is
      // a different bracket, so it carries none.
      record: filtered ? null : trackRecord(st.id),
      active: state.strategy === st.id || (state.strategy === CUSTOM && state.objective === st.id),
    };
  });
  const rs = ruleStrategy();
  const fe = fittedEval();
  const live = fitReady() ? solveByFit() : null;
  rows.push({
    id: RULE,
    label: 'Rule search',
    kind: 'Experimental — fit after the fact, not scored',
    p1: null, ev: null,
    champion: rs ? teams[rs.picks[5][0]] : null,
    filtered: false, record: null,
    active: state.strategy === RULE,
  });
  rows.push({
    id: MODEL,
    label: 'Fitted model',
    kind: 'Evaluated, not selected',
    p1: fe && !fe.stale ? fe.p1 : null,
    ev: fe && !fe.stale ? fe.ev : null,
    scored: !!(fe && !fe.stale),
    stale: !!(fe && fe.stale),
    champion: live ? teams[live[5][0].win] : null,
    filtered: false,
    // Only meaningful for the bracket the evaluation scored, which fittedEval()
    // has just confirmed is the live one.
    record: fe && !fe.stale ? trackRecord(MODEL) : null,
    active: state.strategy === MODEL,
  });
  return rows;
}

/* What the bracket actually did, for a played season: ESPN points against the
 * real outcome, and where that would have finished in the same simulated
 * 30-entry fields P(1st) is measured against (scripts/build_track_record.py).
 * The payload builder embeds it only for exactly the picks this payload
 * carries. Null for a season not yet played or a bracket without a record. */
function trackRecord(id) {
  const tr = state.season && state.season.track_record;
  return tr && tr.strategies && tr.strategies[id] ? tr.strategies[id] : null;
}

function finishText(r) {
  const pool = (state.season.track_record && state.season.track_record.pool_size) || 30;
  return `won ${Math.round(r.won_share * 100)}% of pools · median ${ordinal(r.median_rank)} of ${pool}`;
}

/* Winners the bracket sends through against the seed line, in bracket order,
 * deduplicated: the short answer to "where does this bracket take a stand". */
function chalkDeviations(rounds) {
  const teams = state.season.teams;
  const seen = new Set(), out = [];
  for (const games of rounds) {
    for (const g of games) {
      const w = teams[g.win], l = teams[g.win === g.a ? g.b : g.a];
      if (w.seed > l.seed && !seen.has(g.win)) { seen.add(g.win); out.push(w); }
    }
  }
  return out;
}

function renderHeadline(rounds) {
  const box = document.getElementById('headline');
  if (!box) return;
  const s = state.season;
  const teams = s.teams;
  const champ = teams[rounds[rounds.length - 1][0].win];
  const st = usingOptimized() ? currentStrategy() : null;
  const fe = !st ? fittedEval() : null;
  const nOpp = (s.p1_pool_size || 30) - 1;

  const rr = state.strategy === RULE ? state.rule.result : null;
  const objective = !st ? 'Fitted model'
    : st.id === RULE ? 'Rule search (experimental)'
    : st.id === CUSTOM ? (state.objective === 'ev' ? 'Most expected points, with your filters' : 'Win the pool, with your filters')
    : st.id === 'ev' ? 'Most expected points'
    : 'Win the pool';
  const kind = !st ? 'Evaluated, not selected'
    : st.id === RULE ? 'Experimental, not scored'
    : st.id === CUSTOM ? 'Your pick'
    : st.id === 'ev' ? 'Exact optimum'
    : 'Backtested rule';

  // The estimand, in one sentence, per objective. This is the sentence the
  // page most needed and did not have: what kind of thing the bracket IS.
  const estimand = !st
    ? `The fitted model’s own game-by-game picks. Scored by the same pool referee as the other rows — evaluated, never selected.`
    : st.id === RULE
      ? (rr && rr.mode === 'hand'
        ? `A rule you composed: in each round, every game goes to the team better on one chosen variable. Nothing selected it; whether it reproduces ${ruleTargetText()} in past seasons is shown beside it, and is not evidence about this season.`
        : `A rule found after the fact: in each round, every game goes to the team better on one chosen variable. It was kept because it reproduces ${ruleTargetText()} in ${rr ? ruleYearsText(rr.usedSeasons) : 'the chosen seasons'} — that is what it was searched for, not evidence about this season.`)
    : st.id === CUSTOM
      ? `The best bracket in the candidate pool under your filters, ranked by ${state.objective === 'ev' ? 'expected points' : 'chance of finishing first'}. A belief you supplied, not a validated recommendation.`
    : st.id === 'ev'
      ? `Chosen for the most expected points under the model’s own probabilities — a different objective from winning the pool, and usually a chalkier bracket.`
      : `Pool strategy, not a game-prediction ranking: chosen to maximise the estimated chance of finishing first in a ${nOpp + 1}-entry pool — not the bracket with the most expected points.`;

  // Evidence, stated narrowly. No figures here that could go stale: the
  // audit's numbers live in artifacts/methodology_audit/step18.
  const evidence = !st
    ? `Fitted only on tournaments before ${state.year} — never on this one.`
    : st.id === RULE ? (st.prior && st.prior.m
        ? (rr && rr.mode === 'hand'
          ? `On the ${st.prior.m} played seasons before ${state.year} it reproduces ${ruleTargetText()} in ${st.prior.k}.`
          : `On the ${st.prior.m} played seasons outside that range it reproduces ${ruleTargetText()} in ${st.prior.k}.`)
        : '')
    : st.id === CUSTOM ? ''
    : st.id === 'ev'
      ? `The exact expected-points maximum on this bracket; not a pool backtest.`
      : `Backtested on 15 tournaments (2011–2026, no 2020); the edge over a seed bracket held under an independent market referee.`;

  const nums = st && st.id === RULE
    ? `<span class="hl-num muted">Not scored against the pool: no chance of finishing first, no expected points</span>`
    : st
    ? `<span class="hl-num"><b>${p1Pct(st.p1)}</b> chance of finishing first</span>` +
      `<span class="hl-num"><b>${st.ev.toFixed(0)}</b> expected points</span>`
    : fe && !fe.stale
      ? `<span class="hl-num"><b>${p1Pct(fe.p1)}</b> chance of finishing first</span>` +
        `<span class="hl-num"><b>${fe.ev.toFixed(0)}</b> expected points</span>`
      : `<span class="hl-num muted">${fe && fe.stale ? 'Not scored: the evaluation on file is for a different bracket' : 'Not scored against the pool'}</span>`;

  const dev = chalkDeviations(rounds);
  const devText = dev.length
    ? `Against the seeds: ${dev.slice(0, 4).map(t => `${t.seed} ${t.name}`).join(', ')}${dev.length > 4 ? ` +${dev.length - 4} more` : ''}.`
    : 'Straight chalk: no lower seed advances.';

  const rec = !st ? (fe && !fe.stale ? trackRecord(MODEL) : null)
            : st.id === CUSTOM ? null : trackRecord(st.id);
  const went = rec
    ? `<p class="hl-line hl-went"><b>How it went:</b> ${rec.points.toLocaleString()} points; ${finishText(rec)}
        <span class="hl-evidence">(field median ${rec.pool_median_points.toLocaleString()}, best ${rec.pool_best_points.toLocaleString()}).</span></p>`
    : '';

  box.hidden = false;
  box.innerHTML = `
    <div class="hl-top">
      <span class="hl-season">${state.year} bracket</span>
      <span class="hl-obj">${objective}</span>
      <span class="tag${st && st.id !== CUSTOM && st.id !== RULE ? '' : ' alt'}${st && st.id === RULE ? ' warn-tag' : ''}">${kind}</span>
    </div>
    <div class="champ-card hl-champ">
      <span class="champ-seed">${champ.seed}</span>
      <div class="champ-mid">
        <p class="champ-label">Champion</p>
        <p class="champ-name">${champ.name}</p>
      </div>
      <span class="champ-region">${champ.region}</span>
    </div>
    <div class="hl-nums">${nums}
      <span class="hl-meta">${st && st.id === RULE ? '63 picks · not simulated against any pool' : `63 picks · simulated against ${nOpp} modelled opponents`}</span>
    </div>
    <p class="hl-estimand">${estimand}</p>
    <p class="hl-line">${devText}${evidence ? ` <span class="hl-evidence">${evidence}</span>` : ''}</p>
    ${went}`;
}

function renderCompare() {
  const box = document.getElementById('compare');
  if (!box) return;
  const rows = strategyRows();
  const cell = (v, f) => (v === null || v === undefined ? '<span class="muted">—</span>' : f(v));
  // Played seasons get two more columns: the expectation on the left, the
  // realisation on the right, and the header says which is which.
  const played = rows.some(r => r.record);
  box.hidden = false;
  box.innerHTML = `
    <table class="cmp">
      <thead><tr>
        <th>Strategy</th><th class="num">Chance of 1st</th><th class="num">Exp. points</th><th>Champion</th>
        ${played ? `<th class="num cmp-real">Scored</th><th class="cmp-real">Finish</th>` : ''}
      </tr></thead>
      <tbody>${rows.map(r => `
        <tr class="cmp-row${r.active ? ' on' : ''}" onclick="setStrategy('${r.id}')" role="button" tabindex="0"
            onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setStrategy('${r.id}'); }">
          <td><span class="cmp-label">${r.label}</span>
              <span class="cmp-kind">${r.kind}${r.filtered ? ' · filtered' : ''}${r.stale ? ' · not scored' : ''}</span></td>
          <td class="num">${cell(r.p1, p1Pct)}</td>
          <td class="num">${cell(r.ev, v => v.toFixed(0))}</td>
          <td>${r.champion ? `<span class="cmp-seed">${r.champion.seed}</span> ${r.champion.name}` : '<span class="muted">—</span>'}</td>
          ${played ? `<td class="num cmp-real">${r.record ? r.record.points.toLocaleString() : '<span class="muted">—</span>'}</td>
          <td class="cmp-real cmp-finish">${r.record ? finishText(r.record) : '<span class="muted">—</span>'}</td>` : ''}
        </tr>`).join('')}
      </tbody>
    </table>
    ${played ? `<p class="cmp-foot">Chance and expected points are what the model expected before the tournament;
      Scored and Finish are what happened, against the same simulated 30-entry fields the chance was measured in.
      One season is one draw.</p>` : ''}`;
}


/* ---------- explore a variable ----------
 *
 * EXPLAIN, NOT EDIT. Everything here reads the model and the data; nothing
 * writes to either. The four panels:
 *
 *   Field        percentile rank of every team on the variable (the same
 *                percentileInField() the drawer uses), top of the field and
 *                this bracket's Final Four highlighted
 *   Hinges       the games on THIS board with the largest gap on the
 *                variable, with the model's probability for each -- where
 *                the variable is doing the most, or being overruled
 *   On its own   what the variable predicts by itself, walk-forward: how
 *                often the better-value team won (overall and by round,
 *                with SE), correlation with margin, and a one-variable
 *                fitted model scored on the same held-out seasons as the
 *                full model so the two accuracies are comparable
 *   In the model the variable's coefficient in the full fit with the same
 *                stability and collinearity marks the equation shows
 *
 * Only the Fitted strategy shows the panel, because "in the model" and the
 * per-game probabilities are that model's; the precomputed brackets are not
 * a regression and have no coefficient to explain.
 */
const ROUND_KEYS = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG'];
const ROUND_SHORT = { R64: 'R64', R32: 'R32', S16: 'S16', E8: 'E8', F4: 'F4', NCG: 'Final' };

/* ---------- model sensitivity (Phase B, preregistered) ----------
 *
 * artifacts/methodology_audit/ui_phase_b/PREREGISTRATION_MODEL_SENSITIVITY.md
 * is the definition; fit.js exclusionModels() is the computation; this is
 * the presentation. Per canonical variable: the model refit without it, its
 * held-out accuracy and log loss on the same games as the full model, the
 * probability it assigns to each game on the fitted bracket, and how many
 * of the 63 slots a re-solve under it would change. Computed once per
 * season, on demand, cached in state; consulted by nothing outside the
 * Explore panel.
 */
function sensitivity() {
  if (state.sens && state.sens.year === state.year && state.sens.fitId === state.fit) return state.sens;
  const f = state.fit, src = state.training;
  const models = exclusionModels(src.games, f.cols, src.years, state.year, 2014);
  // The baseline is solved HERE, by the same walk, not taken from whatever
  // render() last put in state.rounds. Both brackets in every "picks
  // changed" count come from solveBracket() in this function, on the same
  // first_round, the same z, the same tie rule, with only the probability
  // function differing -- and nothing random anywhere in that walk.
  const base = solveBracket(winProb);
  const byKey = {};
  models.forEach((m, j) => {
    const key = f.keys[j];
    const keys = f.keys.filter((_, k) => k !== j);
    const cal = m.oos && m.oos.calibration;
    const pFn = (a, b) => {
      const z = state.season.z;
      let t = 0;
      for (let k = 0; k < keys.length; k++) {
        const col = z[keys[k]];
        t += m.fit.beta[k] * (col ? (col[a] || 0) - (col[b] || 0) : 0);
      }
      return winProbFromMargin(t, m.fit.sigma, cal);
    };
    let changed = null;
    if (m.fit.ok) {
      const alt = solveBracket(pFn);
      changed = 0;
      base.forEach((games, r) => games.forEach((g, i) => { if (alt[r][i].win !== g.win) changed++; }));
    }
    byKey[key] = { key, ok: m.fit.ok, oos: m.oos, pFn, changed };
  });
  state.sens = { year: state.year, fitId: f, base, byKey };
  return state.sens;
}

/* SENSITIVITY-COPY-START -- a test forbids causal/importance wording here. */
function sensitivityHTML(meta) {
  const f = state.fit, full = f.oos;
  if (!full) return '';
  const sens = sensitivity();
  const me = sens.byKey[meta.key];
  if (!me || !me.ok || !me.oos) return `<p class="ex-line muted">Not enough history to refit without ${meta.label}.</p>`;
  const pct = v => `${Math.round(v * 100)}%`;
  const ll = o => (o && o.probScore ? o.probScore.logLoss : null);
  const dll = ll(me.oos) !== null && ll(full) !== null ? ll(me.oos) - ll(full) : null;

  // Collinear partner, named in the absorption sentence.
  const i = f.keys.indexOf(meta.key);
  let partner = null;
  if (f.corr) {
    let best = -1, bestAbs = 0;
    for (let j = 0; j < f.corr[i].length; j++) { if (j === i) continue; const a = Math.abs(f.corr[i][j]); if (a > bestAbs) { bestAbs = a; best = j; } }
    if (best >= 0 && bestAbs >= COLLINEAR_R) partner = (state.season.variables.find(v => v.key === f.keys[best]) || {}).label || f.keys[best];
  }

  // Historical: same held-out games, both models.
  const hist = `
    <p class="ex-line">Across the same ${full.n} held-out games (${full.seasons} seasons):
      full model <b>${pct(full.accuracy)}</b> right, log loss ${ll(full) !== null ? ll(full).toFixed(3) : '—'};
      refit excluding ${meta.label} <b>${pct(me.oos.accuracy)}</b> right, log loss ${ll(me.oos) !== null ? ll(me.oos).toFixed(3) : '—'}
      ${dll !== null ? `(Δ log loss <b>${dll >= 0 ? '+' : '−'}${Math.abs(dll).toFixed(3)}</b>, ${dll > 0 ? 'worse without it' : dll < 0 ? 'better without it' : 'no change'})` : ''}.</p>`;

  // Local: the fitted bracket's games, largest |Δp| first.
  const teams = state.season.teams;
  const local = [];
  sens.base.forEach((games, r) => games.forEach(g => {
    const pe = me.pFn(g.a, g.b);
    local.push({ r, g, pe, d: Math.abs(pe - g.p) });
  }));
  local.sort((x, y) => y.d - x.d);
  const localRows = local.slice(0, 5).map(h => {
    const fav = h.g.p >= 0.5 ? h.g.a : h.g.b;
    const pf = fav === h.g.a ? h.g.p : 1 - h.g.p;
    const pe = fav === h.g.a ? h.pe : 1 - h.pe;
    const flips = (pe >= 0.5) !== (pf >= 0.5);
    return `<div class="ex-hinge${flips ? ' over' : ''}">
      <span class="ex-round">${ROUNDS[h.r]}</span>
      <span class="ex-teams"><b>${teams[fav].name}</b> vs ${teams[fav === h.g.a ? h.g.b : h.g.a].name}</span>
      <span class="ex-gap">full model ${pct(pf)}</span>
      <span class="ex-p">refit excluding ${meta.label}: ${pct(pe)}</span>
    </div>`;
  }).join('');

  // Say what the refit actually showed about the collinear partner, not
  // what collinearity usually implies. Measured on the shipped matrix:
  // Overall rating and National rank correlate at 0.99, yet excluding
  // either costs ~+0.085 log loss and excluding both costs the same --
  // the pair is one feature (a rating relative to its rank) that neither
  // carries alone. "Absorbed" would be false there.
  const ABSORB_LL = 0.01;
  const absorb = (partner && dll !== null && Math.abs(dll) < ABSORB_LL)
    ? `${meta.label} moves almost exactly with ${partner} (r ≥ ${COLLINEAR_R}), and the refit without it performs about the same: what it carried was absorbed by ${partner}. A small change does not mean the information carries nothing.`
    : (partner && dll !== null)
      ? `${meta.label} moves almost exactly with ${partner} (r ≥ ${COLLINEAR_R}), yet the refit without it does not recover the full model — together the two carry something the remaining one does not on its own. Read them as a pair, not as two separate quantities; a large change says how the model uses them, not what wins games.`
      : `Excluding a variable lets the ones that move with it absorb it. A small change does not mean the information carries nothing; a large change says how the model uses it, not what wins games.`;

  // Every variable, same two columns, this one highlighted.
  const table = f.keys.map(k => {
    const e = sens.byKey[k];
    const lab = (state.season.variables.find(v => v.key === k) || {}).label || k;
    const d = e && e.ok && e.oos && ll(e.oos) !== null && ll(full) !== null ? ll(e.oos) - ll(full) : null;
    return `<tr class="${k === meta.key ? 'on' : ''}"><td>${lab}</td>
      <td class="num">${e && e.oos ? pct(e.oos.accuracy) : '—'}</td>
      <td class="num">${d !== null ? `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(3)}` : '—'}</td>
      <td class="num">${e && e.changed !== null ? e.changed : '—'}</td></tr>`;
  }).join('');

  return `
    <p class="ex-sub">Each figure is the probability under a model refit without the variable — every other coefficient re-estimated, same training seasons, same held-out folds, same link. It is a refit, not a coefficient set to zero.</p>
    ${hist}
    <p class="ex-sub">On this bracket, the games whose probability moves most under the refit; orange where the favourite would change.</p>
    ${localRows}
    <p class="ex-line">Re-solving the whole bracket under the refit would change <b>${me.changed}</b> of 63 picks. A count only: that bracket is not shown, scored, or ranked.</p>
    <table class="rel-table ex-senstab"><thead><tr><th>Refit excluding</th><th class="num">Held-out right</th><th class="num">Δ log loss</th><th class="num">Picks changed</th></tr></thead><tbody>${table}</tbody></table>
    <p class="ex-sub">${absorb}</p>`;
}
/* SENSITIVITY-COPY-END */

function setExplore(key) {
  state.explore = key;
  renderExplore();
}

function renderExplore() {
  const host = document.getElementById('explore');
  const body = document.getElementById('explore-body');
  if (!host || !body) return;
  const s = state.season;
  if (usingOptimized() || !fitReady() || !s || !state.rounds) { host.hidden = true; return; }
  host.hidden = false;

  const vars = s.variables;
  const meta = vars.find(v => v.key === state.explore) || vars[0];
  const key = meta.key;
  const inModel = CANONICAL_KEYS.indexOf(key) >= 0;

  // --- picker: every variable, grouped, the model's eleven marked ---
  const groups = {};
  for (const v of vars) (groups[v.group] ||= []).push(v);
  const picker = Object.entries(groups).map(([g, vs]) => `
    <div class="ex-group"><span class="ex-gname">${g}</span>
      ${vs.map(v => `<button class="chip${v.key === key ? ' on' : ''}" onclick="setExplore('${v.key}')">
        <span class="chip-name">${v.label}</span>${CANONICAL_KEYS.indexOf(v.key) >= 0 ? '<span class="chip-stat">in model</span>' : ''}
      </button>`).join('')}
    </div>`).join('');

  // --- field ---
  const z = s.z[key] || [], raw = s.raw[key] || [];
  const ranked = s.teams.map((t, i) => ({ i, t, pct: percentileInField(z, raw, i), raw: raw[i] }))
    .filter(r => r.pct !== null).sort((a, b) => b.pct - a.pct);
  const f4 = new Set(state.rounds[3].map(g => g.win));
  const champ = state.rounds[5][0].win;
  const rowHTML = r => `
    <div class="d-row${f4.has(r.i) ? ' lit' : ''}">
      <span class="d-lab"><span class="cmp-seed">${r.t.seed}</span> ${r.t.name}${r.i === champ ? ' <span class="ex-mark">champion</span>' : f4.has(r.i) ? ' <span class="ex-mark">Final Four</span>' : ''}</span>
      <span class="d-track"><i style="left:${Math.max(2, Math.min(98, r.pct))}%"></i></span>
      <span class="d-val">${fmt(r.raw)}</span>
      <span class="d-pct">${ordinal(Math.round(r.pct))}</span>
    </div>`;
  const top = ranked.slice(0, 6);
  const f4rows = ranked.filter(r => f4.has(r.i) && !top.includes(r));
  const field = `
    <p class="g-name">The field on ${meta.label}${meta.higher_better ? '' : ' <span class="d-dir">↓ lower is better</span>'}</p>
    <p class="ex-sub">Percentile among the ${s.teams.length} teams in the ${state.year} tournament, in the better direction — not an absolute scale.</p>
    ${top.map(rowHTML).join('')}
    ${f4rows.length ? `<p class="ex-sub">This bracket’s Final Four, where not already above</p>${f4rows.map(rowHTML).join('')}` : ''}`;

  // --- hinges: largest gaps on THIS board ---
  const gaps = [];
  state.rounds.forEach((games, r) => games.forEach(g => {
    if (raw[g.a] == null || raw[g.b] == null) return;
    const d = z[g.a] - z[g.b];
    const better = d >= 0 ? g.a : g.b, worse = d >= 0 ? g.b : g.a;
    gaps.push({ r, g, gap: Math.abs(d), better, worse, pBetter: better === g.a ? g.p : 1 - g.p });
  }));
  gaps.sort((a, b) => b.gap - a.gap);
  // Literal, not hierarchical: the variable has no "opinion" for the model
  // to overrule. Each line states the gap and which team the full model
  // takes; where that is the team worse on this variable, it says so.
  const hinge = gaps.slice(0, 5).map(h => {
    const other = h.g.win !== h.better;
    return `<div class="ex-hinge${other ? ' over' : ''}">
      <span class="ex-round">${ROUNDS[h.r]}</span>
      <span class="ex-teams"><b>${s.teams[h.better].name}</b> ${meta.higher_better ? 'higher' : 'better'} than ${s.teams[h.worse].name}</span>
      <span class="ex-gap">${h.gap.toFixed(1)}σ apart</span>
      <span class="ex-p">${other ? `full model takes ${s.teams[h.g.win].name}, ${Math.round(100 * (1 - h.pBetter))}%` : `full model takes ${s.teams[h.better].name}, ${Math.round(100 * h.pBetter)}%`}</span>
    </div>`;
  }).join('');

  // --- on its own ---
  const src = state.training;
  const col = src.keys.indexOf(key);
  let own = `<p class="ex-sub muted">Not in the training matrix, so nothing to measure.</p>`;
  let byRound = '';
  if (col >= 0) {
    const rec = variableRecord(src.games, col, state.year);
    const one = fitLinear(src.games, [col], state.year);
    const oneOos = one.ok ? causalWalkForward(src.games, [col], src.years, state.year, 2014) : null;
    const full = state.fit.oos;
    const pct = v => `${Math.round(v * 100)}%`;
    const pm = v => `±${Math.round(v * 100)}`;
    // Exactly what was counted: the team whose PRE-TOURNAMENT value of this
    // variable was better (higher, or lower where lower is better).
    const which = meta.higher_better ? `higher pre-tournament ${meta.label}` : `better (lower) pre-tournament ${meta.label}`;
    const yrs = src.years.filter(y => y < state.year);
    const sample = `<p class="ex-method">Sample: <b>${rec ? rec.n.toLocaleString() : 0}</b> bracket games, ${yrs[0]}–${yrs[yrs.length - 1]} — seasons before ${state.year} only, each measured with information available before that tournament.${oneOos ? ` Held-out evaluation: ${oneOos.seasons} seasons, ${oneOos.n} games.` : ''}</p>`;
    own = rec ? `
      ${sample}
      <p class="ex-line">The team with the ${which} won <b>${pct(rec.betterWins.rate)}</b> <span class="muted">${pm(rec.betterWins.se)}</span> of those games.
        Correlation between the gap on it and the final margin: <b>${rec.corr.toFixed(2)}</b>.</p>
      ${oneOos ? `<p class="ex-line">As a one-variable model, alone: calls <b>${pct(oneOos.accuracy)}</b> of held-out games right
        (log loss ${oneOos.probScore ? oneOos.probScore.logLoss.toFixed(3) : '—'}) —
        the full ${state.fit.keys.length}-variable model calls <b>${pct(full.accuracy)}</b>
        (${full.probScore ? full.probScore.logLoss.toFixed(3) : '—'}) on the same games.</p>` : ''}` : sample + own;
    byRound = rec ? `
      <table class="rel-table ex-rounds"><thead><tr><th></th>${ROUND_KEYS.map(k => `<th class="num">${ROUND_SHORT[k]}</th>`).join('')}</tr></thead>
        <tbody>
          <tr><td class="ex-rowlab">games</td>${ROUND_KEYS.map(k => { const b = rec.byRound[k]; return `<td class="num ex-n">${b ? b.n : '—'}</td>`; }).join('')}</tr>
          <tr><td class="ex-rowlab">won</td>${ROUND_KEYS.map(k => { const b = rec.byRound[k]; return `<td class="num">${b ? `${pct(b.rate)} <span class="rel-se">${pm(b.se)}</span>` : '—'}</td>`; }).join('')}</tr>
        </tbody></table>
      <p class="ex-sub">Descriptive, not a finding: the later rounds are a handful of games a season, and the ± says how little they pin down.</p>` : '';
  }

  // --- in the model ---
  let inm = `<p class="ex-line muted">Not one of the ${CANONICAL_KEYS.length} variables the fitted model uses. Every variable was measured; this one did not earn a place (see the equation’s note).</p>`;
  if (inModel) {
    const i = state.fit.keys.indexOf(key);
    const b = state.fit.beta[i];
    const stab = state.fit.oos && state.fit.oos.stability ? state.fit.oos.stability[i] : null;
    const corr = state.fit.corr;
    let partner = null;
    if (corr) {
      let best = -1, bestAbs = 0;
      for (let j = 0; j < corr[i].length; j++) { if (j === i) continue; const a = Math.abs(corr[i][j]); if (a > bestAbs) { bestAbs = a; best = j; } }
      if (best >= 0 && bestAbs >= COLLINEAR_R) partner = { label: (vars.find(v => v.key === state.fit.keys[best]) || {}).label || state.fit.keys[best], r: corr[i][best], b: state.fit.beta[best] };
    }
    inm = `<p class="ex-line">Weight in the full model: <b>${b < 0 ? '−' : '+'}${Math.abs(b).toFixed(2)}</b> points of margin per standard deviation of edge.
      ${stab && stab.signFlips ? `<span class="ex-warn">Changes sign between held-out seasons (${stab.min.toFixed(1)} to ${stab.max.toFixed(1)}): not readable as a weight on its own.</span>` : ''}
      ${partner ? `<span class="ex-warn">Moves almost exactly with ${partner.label} (r=${partner.r.toFixed(2)}); read the two together: net ${(b + partner.b) < 0 ? '−' : '+'}${Math.abs(b + partner.b).toFixed(2)}.</span>` : ''}</p>`;
  }

  body.innerHTML = `
    <div class="ex-picker">${picker}</div>
    <div class="ex-grid">
      <div class="ex-col">${field}</div>
      <div class="ex-col">
        <p class="g-name">Historical signal, on its own</p>
        ${own}
        <p class="g-name ex-space">Where the gap is largest on this bracket</p>
        <p class="ex-sub">The five games with the largest gap on it, and which team the full model takes.</p>
        ${hinge || '<p class="ex-sub muted">No game on this board separates two teams on it.</p>'}
        ${byRound ? `<p class="g-name ex-space">By round</p>${byRound}` : ''}
        <p class="g-name ex-space">In the full model</p>
        ${inm}
        ${inModel ? `<p class="g-name ex-space">Model sensitivity</p>${sensitivityHTML(meta)}` : ''}
      </div>
    </div>
    <p class="ex-foot">Looking, not editing: the bracket, the probabilities and every number above are the validated model’s and do not change with what is chosen here. The per-variable on/off toggle this replaces was removed because measurement showed choosing the variables bought nothing.</p>`;
}


/* ---------- rule search (experimental strategy) ----------
 *
 * WHAT IT IS. One criterion per round; every game in that round goes to the
 * team better on that one variable. Two modes:
 *
 *   search  The user says which checkpoints a rule must reproduce (any of
 *           the Round of 32 through the champion), over which range of
 *           prior played seasons, from which eligible criteria, using at
 *           most how many distinct criteria; fit.js ruleSearch() returns
 *           every rule that does, simplest first. The first `n` distinct
 *           brackets they give the displayed season are offered, ranked
 *           either simplest first or by how many played seasons OUTSIDE the
 *           fit range the rule also reproduces.
 *   hand    The user composes one criterion per round; the page applies it
 *           and reports which played seasons before this one it reproduces
 *           the checkpoints in. No search, so nothing is selected on.
 *
 * At least one of the Round of 32 through the Final Four must be a
 * checkpoint: without an early one the search cannot prune before the
 * finalists and would have to enumerate keys^4 sequences per season first.
 *
 * Every input above, and which offered bracket is showing, travels in the
 * URL (writeRuleHash), so a searched or composed rule can be sent as a link.
 *
 * WHAT IT IS NOT. Not a model, not validated, not scored: no P(1st), no EV,
 * no track record, never fed to anything. A rule that reproduces the last
 * two seasons is a description of those seasons found after the fact;
 * measured on the shipped data one season needs 2 criteria, two need 3, and
 * three (2024-2026) have none. The page shows that regress rather than
 * hiding it, and shows each searched rule's hit rate on the played seasons
 * outside its fit range -- the one number about it the user cannot tune.
 *
 * WALK-FORWARD, LIKE EVERYTHING ELSE HERE. Fit seasons and hand-mode checks
 * are played seasons strictly before the displayed one; ruleRange() clamps
 * the pickers to that. A 2026 bracket from a rule fit on 2024-2026 would be
 * a bracket fit on its own result.
 */
function playedSeasonsBefore(year) {
  return (state.seasonsIndex || []).filter(x => x.status === 'ready' && x.year < year).map(x => x.year).sort((a, b) => a - b);
}

function ruleSeasonFrom(payload, year) {
  const crit = Object.assign({}, payload.z);
  crit.seed = payload.teams.map(t => -t.seed);
  return { year, first_round: payload.first_round, crit, seed: payload.teams.map(t => t.seed), actual: payload.actual };
}

function ruleLabel(k) {
  return (state.season.variables.find(v => v.key === k) || {}).label || (k === 'seed' ? 'Seed (chalk)' : k);
}
function ruleAllKeys() { return [...Object.keys(state.season.z), 'seed'].sort(); }
/* The by-hand rule as it stands, or its starting point: the season's first
 * listed variable in every round (the overall rating on the shipped data),
 * seed if the season lists none. Never a key the season lacks: an entry the
 * displayed season does not carry (a link from another season, or edited by
 * hand) is replaced by that starting point, so rulePlay() is never handed a
 * criterion it cannot read. */
function ruleHand() {
  const v = state.season.variables;
  const base = v.length && v[0].key in state.season.z ? v[0].key : 'seed';
  if (!state.rule.hand) return Array(ROUNDS.length).fill(base);
  const known = new Set(ruleAllKeys());
  return state.rule.hand.map(k => (known.has(k) ? k : base));
}
/* The eligible criteria: the chosen set restricted to what the displayed
 * season carries, every variable plus seed when nothing (valid) is chosen. */
function ruleKeys() {
  const all = ruleAllKeys();
  const chosen = state.rule.keys ? state.rule.keys.filter(k => all.includes(k)) : [];
  return (chosen.length ? chosen : all).slice().sort();
}
/* Everything the result depends on, in one string: ensureRuleSearch() skips
 * a search whose result is already held, and ruleStrategy() refuses to
 * show a result for inputs no longer on the panel -- including a result
 * computed for a season the page has since moved off. */
function ruleKey() {
  const r = state.rule;
  return JSON.stringify([state.year, r.mode, r.checkpoints, ruleRange().fit, r.n, ruleKeys(), r.rank, r.maxCriteria, r.mode === 'hand' ? ruleHand() : null]);
}

/* The fit range: [from, to] over played seasons strictly before the
 * displayed one. Defaults to the last three. Clamped so it can never include
 * the displayed season -- a bracket from a rule fit on its own result is not
 * something this page will build. */
function ruleRange() {
  const played = playedSeasonsBefore(state.year);
  if (!played.length) return { played, fit: [], from: null, to: null };
  let to = state.rule.to !== null && played.includes(state.rule.to) ? state.rule.to : played[played.length - 1];
  let from = state.rule.from !== null && played.includes(state.rule.from) ? state.rule.from : played[Math.max(0, played.indexOf(to) - 2)];
  if (from > to) [from, to] = [to, from];
  return { played, fit: played.filter(y => y >= from && y <= to), from, to };
}

function ruleTargetText() {
  const cps = new Set(state.rule.checkpoints);
  return RULE_CHECKPOINTS.filter(c => cps.has(c.r)).map(c => c.label.toLowerCase()).join(', ').replace(/, ([^,]*)$/, ' and $1');
}
function ruleYearsText(years) {
  if (!years || !years.length) return '';
  return years.length === 1 ? String(years[0]) : `${years[0]}–${years[years.length - 1]}`;
}

/* The criterion that decided round r under the rule on screen, for the
 * board's column header. Empty unless the rule strategy is showing. */
function ruleRoundLabel(r, sep) {
  if (state.strategy !== RULE) return '';
  const st = ruleStrategy();
  if (!st || !st.rule) return '';
  const k = st.rule[Math.min(r, st.rule.length - 1)];
  return sep ? `${sep}${ruleLabel(k)}` : `<span class="r-sub">${ruleLabel(k)}</span>`;
}

/* Fetch what the search needs, run it, keep the offered brackets with their
 * outside-the-range hit rate. Cached on state.rule.result by every input. */
const RULE_RANK_POOL = 500;   // distinct brackets scored when ranking by outside-range hits

async function ensureRuleSearch() {
  const s = state.season;
  if (!s || s.status !== 'ready') return;
  const { played, fit: fitYears } = ruleRange();
  const key = ruleKey();
  if (state.rule.result && state.rule.result.key === key) return;
  // A new list starts at its first bracket -- unless this is the first
  // search of the page, where `chosen` may have come from a shared link
  // and names the bracket that link promised.
  if (state.rule.result && state.rule.chosen) { state.rule.chosen = 0; writeHash(); }
  // Every control calls this without awaiting, so two quick clicks overlap
  // here. Only the latest call may write a result: an earlier one finishing
  // last would leave a result for inputs no longer on the panel.
  const token = ++state.rule.token;
  state.rule.busy = true; renderRulePanel();
  try {
    const payloads = {};
    for (const y of played) payloads[y] = ruleSeasonFrom(await loadSeason(y), y);
    if (token !== state.rule.token) return;
    const complete = y => payloads[y].actual && payloads[y].actual.every(r => r.length);
    const here = ruleSeasonFrom(s, state.year);
    const cps = state.rule.checkpoints;
    if (state.rule.mode === 'hand') {
      const seq = ruleHand();
      const rounds = ruleBracket(here, seq);
      const per = played.filter(complete).map(y => ({ year: y, ok: ruleReproduces(payloads[y], seq, cps) }));
      state.rule.result = { key, mode: 'hand', brackets: [{ seq, rounds, picks: rounds.map(g => g.map(x => x.win)), complexity: ruleComplexity(seq), per }] };
      // No early return: a `return` here used to leave the try block and
      // skip the redraw below, so hand mode sat on "Searching…" with its
      // result held in state and the board empty.
    } else {
      const fit = fitYears.filter(complete).map(y => payloads[y]);
      const keys = ruleKeys();
      const res = fit.length ? ruleSearch(fit, keys, new Set(cps)) : { rules: [], usedSeasons: [], backedOff: true };
      // "Outside" means outside the seasons the rule was actually selected on:
      // after a back-off the dropped seasons count too, and count as misses.
      const outside = played.filter(y => complete(y) && !res.usedSeasons.includes(y)).map(y => payloads[y]);
      const brackets = [], seen = new Set();
      const cap = state.rule.rank === 'outside' ? Math.max(RULE_RANK_POOL, state.rule.n) : state.rule.n;
      // ruleSearch() orders by distinct criteria first, so the first rule over
      // the criteria cap means every later one is too: stop there.
      const maxCx = state.rule.maxCriteria;
      let overCap = false;
      for (const seq of res.rules) {
        const complexity = ruleComplexity(seq);
        if (maxCx !== null && complexity[0] > maxCx) { overCap = true; break; }
        const rounds = ruleBracket(here, seq);
        const sig = rounds.map(g => g.map(x => x.win).join(',')).join('|');
        if (seen.has(sig)) continue;
        seen.add(sig);
        const hits = outside.filter(p => ruleReproduces(p, seq, cps)).map(p => p.year);
        brackets.push({ seq, rounds, picks: rounds.map(g => g.map(x => x.win)), complexity, outside: { k: hits.length, m: outside.length, years: hits } });
        if (brackets.length >= cap) break;
      }
      const scored = brackets.length;
      if (state.rule.rank === 'outside') brackets.sort((a, b) => b.outside.k - a.outside.k || a.complexity[0] - b.complexity[0] || a.complexity[1] - b.complexity[1]);
      state.rule.result = { key, mode: 'search', brackets: brackets.slice(0, state.rule.n), scored, usedSeasons: res.usedSeasons, requested: fitYears, backedOff: res.backedOff, nRules: res.rules.length, lastRound: res.lastRound, nOutside: outside.length, overCap };
    }
  } finally {
    if (token === state.rule.token) state.rule.busy = false;
  }
  if (token !== state.rule.token) return;
  renderStrategies();
  render();
}

/* The chosen rule bracket in the shape currentStrategy() returns for the
 * precomputed strategies, so solveFromPicks(), the headline, the table and
 * the exporter all take it through the paths they already have. p1 and ev
 * are deliberately absent. */
function ruleStrategy() {
  const r = state.rule.result;
  if (!r || !r.brackets.length || !state.season || state.season.status !== 'ready') return null;
  // A result is only ever shown for the inputs it was computed from. The
  // season changing underneath it used to leave the previous season's picks
  // -- team indices into a different bracket -- on the new season's board.
  if (r.key !== ruleKey()) return null;
  const i = Math.min(state.rule.chosen, r.brackets.length - 1);
  const b = r.brackets[i];
  const seqText = b.seq.map(ruleLabel).join(' → ') + (b.seq.length < 6 ? ' (later rounds reuse the last criterion)' : '');
  const cx = `${b.complexity[0]} ${b.complexity[0] === 1 ? 'criterion' : 'criteria'}, ${b.complexity[1]} ${b.complexity[1] === 1 ? 'switch' : 'switches'}`;
  let note, prior;
  if (r.mode === 'hand') {
    const ok = b.per.filter(x => x.ok).map(x => x.year);
    note = `Rule, composed by hand: ${seqText}. ${cx}. Reproduces ${ruleTargetText()} in ${ok.length} of ${b.per.length} played seasons before ${state.year}${ok.length ? ` (${ok.join(', ')})` : ''}.`;
    prior = { k: ok.length, m: b.per.length };
  } else {
    note = `Rule: ${seqText}. Reproduces ${ruleTargetText()} in ${ruleYearsText(r.usedSeasons)}${r.backedOff ? ` — the longest range with any surviving rule; ${ruleYearsText(r.requested)} has none` : ''}. ` +
      `${cx}; ${r.nRules.toLocaleString()} rules survived, this is bracket ${i + 1} of ${r.brackets.length} offered.`;
    prior = { k: b.outside.k, m: b.outside.m };
  }
  return { id: RULE, label: 'Rule search', note, picks: b.picks, rule: b.seq, prior };
}

/* Every control ends here: the inputs go into the URL (they are part of the
 * selection a link carries, see writeRuleHash) and the search runs if they
 * changed. */
function ruleChanged() { writeHash(); ensureRuleSearch(); }
function setRuleMode(m) { state.rule.mode = m; if (m === 'hand') state.rule.hand = ruleHand(); ruleChanged(); }
function setRuleCheckpoint(r, on) {
  const cps = new Set(state.rule.checkpoints);
  if (on) cps.add(r); else cps.delete(r);
  // At least one early checkpoint (Round of 32 through Final Four): without
  // one the search cannot prune before the finalists and would enumerate
  // keys^4 sequences per season before its first constraint.
  if (!RULE_EARLY_ROUNDS.some(x => cps.has(x))) { renderRulePanel(); return; }
  state.rule.checkpoints = [...cps].sort((a, b) => a - b);
  ruleChanged();
}
function setRuleRange(from, to) { state.rule.from = from; state.rule.to = to; ruleChanged(); }
function setRuleLast(n) { const p = playedSeasonsBefore(state.year); setRuleRange(p[Math.max(0, p.length - n)], p[p.length - 1]); }
function setRuleN(n) { n = Math.max(1, Math.min(RULE_N_MAX, Math.round(Number(n) || 5))); state.rule.n = n; ruleChanged(); }
function setRuleRank(r) { state.rule.rank = r; ruleChanged(); }
function setRuleMax(m) { state.rule.maxCriteria = m === null ? null : Math.max(1, Math.min(ROUNDS.length, Math.round(Number(m)))); ruleChanged(); }
function setRuleKey(k, on) {
  const cur = new Set(ruleKeys());
  if (on) cur.add(k); else cur.delete(k);
  if (!cur.size) return;
  state.rule.keys = [...cur].sort();
  ruleChanged();
}
function setRuleKeysAll(on) { state.rule.keys = on ? null : ['seed']; ruleChanged(); }
function setRuleHand(r, k) { const h = ruleHand().slice(); h[r] = k; state.rule.hand = h; ruleChanged(); }
function setRuleChosen(i) { state.rule.chosen = Math.max(0, i); writeHash(); renderStrategies(); render(); }

/* RULE-COPY-START -- the wording guard scans this renderer too. */
function renderRulePanel() {
  const host = document.getElementById('rulepanel');
  const body = document.getElementById('rule-body');
  if (!host || !body) return;
  if (state.strategy !== RULE || !state.season || state.season.status !== 'ready') { host.hidden = true; return; }
  host.hidden = false; host.open = true;
  // Only a result for the inputs on the panel; a held one for other inputs
  // (or another season) is not shown while the new search runs.
  const r = state.rule.result && state.rule.result.key === ruleKey() ? state.rule.result : null;
  const { played, fit, from, to } = ruleRange();
  const cps = new Set(state.rule.checkpoints);
  const keysOn = new Set(ruleKeys());
  const all = ruleAllKeys();
  const groups = {};
  for (const v of state.season.variables) (groups[v.group] ||= []).push(v.key);
  groups['Seed'] = ['seed'];

  const modes = `
    <div class="rule-group"><span class="ex-gname">Mode</span>
      <button class="chip${state.rule.mode === 'search' ? ' on' : ''}" onclick="setRuleMode('search')"><span class="chip-name">Search past seasons</span></button>
      <button class="chip${state.rule.mode === 'hand' ? ' on' : ''}" onclick="setRuleMode('hand')"><span class="chip-name">Compose by hand</span></button>
    </div>`;
  const checkpoints = `
    <div class="rule-group"><span class="ex-gname">Must reproduce</span>
      ${RULE_CHECKPOINTS.map(c => `<label class="rule-check"><input type="checkbox" ${cps.has(c.r) ? 'checked' : ''} onchange="setRuleCheckpoint(${c.r}, this.checked)"> ${c.label}</label>`).join('')}
      <span class="ex-sub">at least one of Round of 32 through Final Four</span>
    </div>`;
  const yearOpts = sel => played.map(y => `<option value="${y}"${y === sel ? ' selected' : ''}>${y}</option>`).join('');
  const range = `
    <div class="rule-group"><span class="ex-gname">Fit seasons</span>
      <select class="rule-select" onchange="setRuleRange(Number(this.value), ${to})">${yearOpts(from)}</select>
      <span class="ex-sub">to</span>
      <select class="rule-select" onchange="setRuleRange(${from}, Number(this.value))">${yearOpts(to)}</select>
      <button class="chip" onclick="setRuleLast(3)"><span class="chip-name">last 3</span></button>
      <button class="chip" onclick="setRuleLast(4)"><span class="chip-name">last 4</span></button>
      <span class="ex-sub">played seasons before ${state.year} only; ${fit.length} in range, ${played.length - fit.length} outside it</span>
    </div>`;
  const nAndRank = `
    <div class="rule-group"><span class="ex-gname">Offer</span>
      <input class="rule-num" type="number" min="1" max="${RULE_N_MAX}" value="${state.rule.n}" onchange="setRuleN(this.value)"> brackets,
      <button class="chip${state.rule.rank === 'simple' ? ' on' : ''}" onclick="setRuleRank('simple')"><span class="chip-name">simplest first</span></button>
      <button class="chip${state.rule.rank === 'outside' ? ' on' : ''}" onclick="setRuleRank('outside')"><span class="chip-name">most seasons outside the range first</span></button>
      ${state.rule.rank === 'outside' ? `<span class="ex-sub">ranked among the ${RULE_RANK_POOL} simplest distinct brackets</span>` : ''}
    </div>
    <div class="rule-group"><span class="ex-gname">Criteria per rule</span>
      <button class="chip${state.rule.maxCriteria === null ? ' on' : ''}" onclick="setRuleMax(null)"><span class="chip-name">any</span></button>
      ${[1, 2, 3].map(m => `<button class="chip${state.rule.maxCriteria === m ? ' on' : ''}" onclick="setRuleMax(${m})"><span class="chip-name">at most ${m}</span></button>`).join('')}
      <span class="ex-sub">distinct variables a rule may switch between across the rounds</span>
    </div>`;
  const vars = `
    <details class="rule-vars"><summary>Eligible criteria: ${keysOn.size} of ${all.length}
      <button class="chip" onclick="event.preventDefault(); setRuleKeysAll(true)"><span class="chip-name">all</span></button>
      <button class="chip" onclick="event.preventDefault(); setRuleKeysAll(false)"><span class="chip-name">seed only</span></button></summary>
      ${Object.entries(groups).map(([g, ks]) => `<div class="rule-group"><span class="ex-gname">${g}</span>
        ${ks.map(k => `<label class="rule-check"><input type="checkbox" ${keysOn.has(k) ? 'checked' : ''} onchange="setRuleKey('${k}', this.checked)"> ${ruleLabel(k)}</label>`).join('')}</div>`).join('')}
    </details>`;
  const handOpts = i => Object.entries(groups).map(([g, ks]) => `<optgroup label="${g}">${ks.map(k =>
    `<option value="${k}"${ruleHand()[i] === k ? ' selected' : ''}>${ruleLabel(k)}</option>`).join('')}</optgroup>`).join('');
  const hand = `
    <div class="rule-hand">${ROUNDS.map((rn, i) => `<label class="rule-handrow"><span class="ex-round">${rn}</span>
      <select class="rule-select" onchange="setRuleHand(${i}, this.value)">${handOpts(i)}</select></label>`).join('')}
    </div>`;

  let results = '';
  if (state.rule.busy) results = `<p class="ex-line">Searching…</p>`;
  else if (!r) results = '';
  else if (r.mode === 'hand') {
    const b = r.brackets[0];
    results = `<p class="ex-line">This rule gives ${state.year} a bracket with champion <b>${state.season.teams[b.picks[b.picks.length - 1][0]].name}</b>. ${b.per.length
      ? `Across the ${b.per.length} played seasons before ${state.year} it reproduces ${ruleTargetText()} in:
      ${b.per.map(x => `<span class="rule-yr${x.ok ? ' ok' : ''}">${x.year}</span>`).join(' ')}`
      : `There are no played seasons before ${state.year} to check it against.`}</p>`;
  } else if (!played.length) {
    results = `<p class="ex-line">No played seasons before ${state.year}: there is nothing to search. A rule can still be composed by hand.</p>`;
  } else if (!r.brackets.length && r.overCap) {
    results = `<p class="ex-line">${r.nRules.toLocaleString()} rules reproduce ${ruleTargetText()} in ${ruleYearsText(r.usedSeasons)}, but none using at most ${state.rule.maxCriteria} ${state.rule.maxCriteria === 1 ? 'criterion' : 'criteria'}. Raise the cap to see them.</p>`;
  } else if (!r.brackets.length) {
    results = `<p class="ex-line">No single-criterion rule reproduces ${ruleTargetText()} in ${ruleYearsText(r.requested)} — nor in any shorter range ending in ${r.requested[r.requested.length - 1] || ''}. Fewer checkpoints, more criteria, or a different range may have one.</p>`;
  } else {
    const capText = state.rule.maxCriteria === null ? '' : ` (those using at most ${state.rule.maxCriteria} ${state.rule.maxCriteria === 1 ? 'criterion' : 'criteria'})`;
    const head = r.backedOff
      ? `<p class="ex-line"><b>No rule reproduces ${ruleTargetText()} across ${ruleYearsText(r.requested)}.</b> The longest range ending in ${r.requested[r.requested.length - 1]} with a surviving rule is ${ruleYearsText(r.usedSeasons)}: ${r.nRules.toLocaleString()} rules; ${r.brackets.length} of the ${r.scored} distinct brackets they${capText} give ${state.year} are offered below.</p>`
      : `<p class="ex-line">${r.nRules.toLocaleString()} rules reproduce ${ruleTargetText()} in every season of ${ruleYearsText(r.usedSeasons)}; ${r.brackets.length} of the ${r.scored} distinct brackets they${capText} give ${state.year} are offered below.</p>`;
    const list = r.brackets.map((b, i) => `
      <button class="rule-row${i === Math.min(state.rule.chosen, r.brackets.length - 1) ? ' on' : ''}" onclick="setRuleChosen(${i})">
        <span class="rule-seq">${b.seq.map(ruleLabel).join(' → ')}</span>
        <span class="rule-meta">${b.complexity[0]} ${b.complexity[0] === 1 ? 'criterion' : 'criteria'} · ${b.complexity[1]} ${b.complexity[1] === 1 ? 'switch' : 'switches'} · champion ${state.season.teams[b.picks[5][0]].name}
          · outside the range: ${b.outside.m ? `${b.outside.k} of ${b.outside.m} seasons${b.outside.k ? ` (${b.outside.years.join(', ')})` : ''}` : 'no other seasons to check'}</span>
      </button>`).join('');
    results = head + list;
  }
  body.innerHTML = `<div class="rule-controls">${modes}${checkpoints}${state.rule.mode === 'hand' ? hand : range + nAndRank + vars}</div>` + results + `
    <p class="ex-foot">Experimental. Searched rules were found by looking for what reproduces past results, which is why they reproduce them; the seasons outside the range each rule also reproduces is the only number here it was not chosen on. Nothing on this panel is scored against the pool or used by any other part of the page.</p>`;
}
/* RULE-COPY-END */

/* ---------- narrow-viewport round navigation ----------
 *
 * See the @media (max-width: 720px) rule in app.css: below that width, .board
 * shows one .round at a time (whichever carries the `active` class) instead
 * of scrolling six columns sideways. Every round is already built by
 * render() regardless of viewport, so stepping between them here only
 * toggles a class -- it never re-solves the bracket, and it is a correct
 * no-op on a wide viewport where the CSS rule does not apply and every round
 * is visible no matter which one is `active`.
 */
function updateMobileNav() {
  const nav = document.getElementById('board-nav');
  const dots = document.getElementById('rnav-dots');
  const label = document.getElementById('rnav-label');
  if (!nav || !dots || !label) return;
  if (!state.rounds) { nav.hidden = true; dots.hidden = true; return; }
  nav.hidden = false;
  dots.hidden = false;
  label.textContent = `${ROUNDS[state.mobileRound]}${ruleRoundLabel(state.mobileRound, ' · ')} · ${state.mobileRound + 1}/${ROUNDS.length}`;
  document.getElementById('rnav-prev').disabled = state.mobileRound === 0;
  document.getElementById('rnav-next').disabled = state.mobileRound === ROUNDS.length - 1;
  dots.innerHTML = ROUNDS.map((r, ri) => `
    <button class="rnav-dot${ri === state.mobileRound ? ' on' : ''}" aria-label="${r}"
            onclick="jumpMobileRound(${ri})"></button>`).join('');
}

function jumpMobileRound(r) {
  if (!state.rounds || r < 0 || r >= ROUNDS.length || r === state.mobileRound) return;
  state.mobileRound = r;
  document.querySelectorAll('#board > .round').forEach(el => {
    el.classList.toggle('active', Number(el.dataset.r) === r);
  });
  updateMobileNav();
}

function setMobileRound(delta) {
  if (!state.rounds) return;
  jumpMobileRound(Math.max(0, Math.min(ROUNDS.length - 1, state.mobileRound + delta)));
}

/* Swipe left/right on the board to step a round, on top of the arrows and
 * dots. Bound once, unconditionally -- gated at touchend time by matchMedia
 * rather than only attaching the listener under the breakpoint, because
 * @media alone does not fire JS when a viewport crosses it. Checked at
 * touchend rather than touchstart so rotating mid-gesture cannot fire a jump
 * the layout it lands on was never meant to react to.
 *
 * WIDE VIEWPORTS ARE DELIBERATELY LEFT ALONE. There .board scrolls
 * horizontally by design, and a drag across it is that scroll, not a page
 * turn -- reinterpreting it here would fight the browser's own gesture. */
function wireBoardSwipe() {
  const board = document.getElementById('board');
  if (!board) return;
  const SWIPE_PX = 40;
  let x0 = null;
  board.addEventListener('touchstart', e => { x0 = e.touches[0].clientX; }, { passive: true });
  board.addEventListener('touchend', e => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0;
    x0 = null;
    if (!window.matchMedia('(max-width: 720px)').matches) return;
    if (dx <= -SWIPE_PX) setMobileRound(1);
    else if (dx >= SWIPE_PX) setMobileRound(-1);
  }, { passive: true });
}

/* The fitted model, written out.
 *
 * Terms are ordered by magnitude rather than by menu position, so the variables
 * actually carrying the model come first. Every delta is a difference in
 * standard deviations between the two teams, which is why a coefficient reads as
 * POINTS OF MARGIN per standard deviation of edge.
 */
// r at or above this is "the same signal, not two effects" -- see
// pairwiseCorrelations() in fit.js. 0.8 is a conventional high-collinearity
// line in applied regression; the canonical set's actual worst offender
// (Overall rating vs National rank) sits at ~0.99, well past it either way.
const COLLINEAR_R = 0.8;

function equationHTML() {
  const f = state.fit;
  const label = Object.fromEntries(state.season.variables.map(v => [v.key, v.label]));

  // A coefficient that changes sign between walk-forward folds is not a
  // finding, however large it looks. Marking those is the difference between
  // showing the model and vouching for it.
  const stab = f.oos ? f.oos.stability : null;

  // A coefficient can be perfectly sign-stable across every fold and STILL
  // not mean what it looks like, if the column it belongs to is nearly a
  // duplicate of another enabled one: stability() cannot see that, because
  // both of a collinear pair's coefficients agree with themselves fold to
  // fold -- they just disagree with each other about how to split one
  // shared signal. `i` here is the position in f.keys/f.cols/f.corr, kept on
  // each term so the lookup survives the magnitude sort below.
  const corr = f.corr;
  const partnerOf = i => {
    if (!corr) return null;
    let best = -1, bestAbs = 0;
    for (let j = 0; j < corr[i].length; j++) {
      if (j === i) continue;
      const a = Math.abs(corr[i][j]);
      if (a > bestAbs) { bestAbs = a; best = j; }
    }
    return best >= 0 && bestAbs >= COLLINEAR_R ? { j: best, r: corr[i][best] } : null;
  };

  const terms = f.keys
    .map((k, i) => ({ i, k, b: f.beta[i], label: label[k] || k, s: stab ? stab[i] : null, partner: partnerOf(i) }))
    .sort((a, b) => Math.abs(b.b) - Math.abs(a.b));

  const body = terms.map((t, i) => {
    const sign = t.b < 0 ? '\u2212' : '+';
    const mag = Math.abs(t.b).toFixed(2);
    const weak = Math.abs(t.b) < 0.05;
    const shaky = t.s && t.s.signFlips;
    const collinear = t.partner !== null;
    const cls = weak ? ' weak' : [shaky && ' shaky', collinear && ' collinear'].filter(Boolean).join('');
    const tips = [];
    if (weak) {
      tips.push('Essentially zero weight');
    } else {
      if (shaky) {
        tips.push(`Unstable: ranged ${t.s.min.toFixed(1)} to ${t.s.max.toFixed(1)} across held-out seasons, changing sign. Do not read this number as a stable weight.`);
      }
      if (collinear) {
        const partnerLabel = label[f.keys[t.partner.j]] || f.keys[t.partner.j];
        const partnerB = f.beta[t.partner.j];
        const net = t.b + partnerB;
        const netSign = net < 0 ? '\u2212' : '+';
        tips.push(
          `Moves almost exactly with \u0394${partnerLabel} (r=${t.partner.r.toFixed(2)}, over the seasons this fit trained on). `
          + `Their coefficients split credit for one shared signal, not two independent quantities \u2014 `
          + `read them together: ${sign}${mag} ${partnerB < 0 ? '\u2212' : '+'} ${Math.abs(partnerB).toFixed(2)} = ${netSign}${Math.abs(net).toFixed(2)}, not this number alone.`
        );
      }
    }
    const marks = weak ? '' : `${shaky ? '<i class="warn" aria-label="unstable">*</i>' : ''}${collinear ? '<i class="warn collinear-mark" aria-label="collinear">\u2020</i>' : ''}`;
    return `<span class="term${cls}" title="${tips.join(' ')}">` +
           `${i === 0 && t.b >= 0 ? '' : `<i class="op">${sign}</i>`}` +
           `<b>${mag}</b><span class="dv">\u0394${t.label}</span>` +
           `${marks}</span>`;
  }).join('');

  const nShaky = terms.filter(t => t.s && t.s.signFlips).length;
  const nCollinear = terms.filter(t => t.partner !== null).length;

  return `
    <div class="eq">
      <p class="eq-head">
        <span class="eq-lhs">team A beats team B by</span>
        <span class="eq-eq">=</span>
      </p>
      <p class="eq-body">${body}<span class="term unit">points</span></p>
      <p class="eq-foot">
        \u0394 is team A minus team B, in standard deviations within the season,
        so each number is points of margin per standard deviation of edge.
        Every \u0394 points the same way: stats where a lower raw number is better
        (Defense is points allowed, National rank is a rank) are flipped before
        standardising, so a positive weight always means more of the good thing.
        No intercept: swapping the teams flips the sign exactly.
        A positive margin is the pick; typical error is
        \u00b1${f.oos ? f.oos.mae.toFixed(1) : f.sigma.toFixed(1)} points.
      </p>
      ${nShaky ? `<p class="eq-warn">
        <i class="warn">*</i> ${nShaky} of these ${terms.length} coefficients change sign
        between held-out seasons. The equation as a whole still predicts \u2014
        that is what the out-of-sample figure measures \u2014 but those individual
        numbers are splitting credit between variables that overlap, and are
        not readable as "what this variable is worth".
      </p>` : ''}
      ${nCollinear ? `<p class="eq-warn">
        <i class="warn collinear-mark">\u2020</i> ${nCollinear} of these ${terms.length} coefficients belong to a
        pair that moves together (r \u2265 ${COLLINEAR_R}) most seasons. A big number on one of a
        collinear pair and an opposite big number on the other is not two quantities
        pulling apart \u2014 it is one signal, split two ways. Hover a marked term for
        its partner and their combined weight.
      </p>` : ''}
      ${nShaky || nCollinear ? `<p class="eq-warn">
        The variable set is fixed because dropping the redundant ones was measured
        and did not predict any better \u2014 it only made the coefficients easier to read.
      </p>` : ''}
      ${reliabilityHTML(f.oos)}
    </div>`;
}

/* Does "70%" come true 70% of the time? The check the accuracy figure cannot
 * make -- see reliabilityTable() in fit.js. Rendered from the walk-forward
 * held-out games strictly before the displayed season, under the same link
 * the board's percentages use, so it is evidence about THESE numbers. */
function reliabilityHTML(o) {
  if (!o || !o.reliability) return '';
  const bins = o.reliability.filter(b => b.n > 0);
  if (!bins.length) return '';
  const pct = v => `${Math.round(v * 100)}%`;
  const rows = bins.map(b => {
    const gap = (b.actual - b.predicted) / b.se;
    // Two sigma is flagged, not judged: with five bins one of them sitting
    // near two sigma is roughly what chance produces, and the 0.5-0.6 bin in
    // particular moves with its own edge (fit.js, calibrate()'s comment).
    const cls = Math.abs(gap) >= 2 ? ' off' : '';
    return `<tr class="rel-row${cls}">
      <td>${pct(b.lo)}\u2013${pct(b.hi)}</td>
      <td class="num">${b.n}</td>
      <td class="num">${pct(b.predicted)}</td>
      <td class="num">${pct(b.actual)} <span class="rel-se">\u00b1${Math.round(b.se * 100)}</span></td>
    </tr>`;
  }).join('');
  const n = bins.reduce((s, b) => s + b.n, 0);
  return `
    <div class="rel">
      <p class="rel-head">Does the percentage mean what it says?
        <span class="rel-sub">${n} held-out games from seasons before ${state.year}, grouped by how confident the model was in the favourite.</span></p>
      <table class="rel-table">
        <thead><tr><th>Model said</th><th class="num">Games</th><th class="num">Avg. said</th><th class="num">Favourite actually won</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="rel-foot">
        \u00b1 is what chance alone would move the observed rate by if the model's
        percentage were exactly right. The lowest band is a known weak spot in the
        reading rather than the model: it is coin-flip games, and which side of the
        50% line a handful of them fall on shifts it by several points.
      </p>
    </div>`;
}

/* Colour is about SLOT correctness: did the model put this team in this game?
 *
 *   green   the model has the right team here
 *   red     the model has the wrong team here; the one that belongs is named
 *           beneath it, struck through
 *   plain   nothing to grade -- the Round of 64 is given rather than predicted,
 *           and once reality leaves a branch its later slots never existed
 *
 * "Picked" (advanced by this bracket) stays a separate signal from "correct",
 * because a bold pick that came off and a safe pick that came off should not
 * look the same as each other, nor as a miss.
 */
function gameHTML(g, round, actualGame) {
  const pa = g.p === undefined ? null : g.p;
  return `
    <div class="game">
      ${sideHTML(g.a, g.win === g.a, pa === null ? null : pa, g.b, round, actualGame ? actualGame.a : null)}
      ${sideHTML(g.b, g.win === g.b, pa === null ? null : 1 - pa, g.a, round, actualGame ? actualGame.b : null)}
    </div>`;
}

/* The percentage badge always comes from the live fitted model (fit.js), even
 * under a precomputed strategy that picked its winners a different way -- see
 * fitReady(). The two cases need different wording: under the Fitted strategy
 * this number IS the pick; under the others it is a second opinion that can
 * legitimately disagree with the bracket on screen. */
function probTitle() {
  return usingOptimized()
    ? "The live fitted model's estimated chance of winning this game — a separate estimate from the one that built this bracket, so it can differ from the pick shown"
    : "The fitted model's estimated chance of winning this game";
}

function sideHTML(i, picked, p, oppI, round, actualHere) {
  const t = state.season.teams[i];
  const upset = picked && state.season.teams[oppI].seed < t.seed;

  // The Round of 64 field is fixed, so being "in" it is not a prediction.
  const gradeable = round > 0 && actualHere !== null && actualHere !== undefined;
  const right = gradeable && actualHere === i;
  const wrong = gradeable && actualHere !== i;
  const should = wrong ? state.season.teams[actualHere] : null;

  return `
    <button class="side${picked ? ' picked' : ''}${upset ? ' upset' : ''}` +
    `${right ? ' right' : ''}${wrong ? ' wrong' : ''}" onclick="openTeam(${i})">
      <span class="seed">${t.seed}</span>
      <span class="tcol">
        <span class="tname">${t.name}</span>
        ${should ? `<span class="should" title="Actually reached this game">${should.name}</span>` : ''}
      </span>
      ${upset ? '<span class="badge up" title="Lower seed picked">UPSET</span>' : ''}
      ${p === null ? '' : `<span class="sc" title="${probTitle()}">${Math.round(p * 100)}%</span>`}
    </button>`;
}

/* ---------- weights ---------- */

/* The strategy picker. Replaces the old variable grid.

 * ORDERED BY THE OBJECTIVE, NOT BY SCORE, because the objectives are not
 * comparable to each other: the P(1st) bracket is supposed to look worse on
 * expected points and the expected-points bracket is supposed to look worse on
 * P(1st). Sorting by either number would imply one of them is losing.
 */
function renderStrategies() {
  const s = state.season;
  if (!s || s.status !== 'ready') return;

  // "backtested" and "exact" are different claims and the tag should not blur
  // them. The win-maximising rule has no closed form -- P(1st) depends on the
  // whole opponent field -- so its evidence is out-of-sample performance across
  // 15 seasons. Expected points does have a closed form and is solved exactly,
  // which is a stronger claim about this bracket and a weaker one about pools.
  // WITH FILTERS ACTIVE THE CARDS SHOW THE FILTERED SCORES, so each card
  // answers "what would I get if I asked THIS question of the brackets I have
  // narrowed to". Showing the unfiltered figures instead made the card and the
  // board disagree -- 941 on the card against 917 on the board -- which reads as
  // one of them being wrong rather than as two different scopes.
  const filt = state.strategy === MODEL ? null : filteredEntry().entry;
  const opts = (s.strategies || []).map(st => ({
    id: st.id, label: st.label, sub: st.note,
    tag: st.id === 'ev' ? 'exact' : 'backtested',
    // With filters active the strategy cards show which QUESTION is being
    // asked, so the objective stays lit rather than every card going dark.
    active: state.strategy === st.id || (state.strategy === CUSTOM && state.objective === st.id),
    stat: (() => {
      const v = (filt && filt.by && filt.by[st.id]) || st;
      return `${p1Pct(v.p1)} to win · ${v.ev.toFixed(0)} pts`
           + (filt ? ' · filtered' : '');
    })(),
  }));
  opts.push({
    id: RULE,
    label: 'Rule search',
    sub: 'Experimental. One criterion per round — every game in that round goes to the team '
       + 'better on one variable — searched for because it reproduces the checkpoints you pick over '
       + 'played seasons before this one, or composed by hand. Found after the fact, not validated, '
       + 'and not scored against the pool: no chance of finishing first, no expected points.',
    tag: 'experimental',
    stat: '',
  });
  opts.push({
    id: MODEL,
    label: 'Fitted model',
    // Used to end "...This is the only strategy the variable weights apply
    // to" -- a control that was removed 2026-08-29 (see the file header: it
    // measured null). What actually still sets this card apart from the
    // other two is the equation printed below it (equationHTML(), only
    // rendered while this strategy is active), so the sentence now says that
    // instead of describing a control nobody can find on the page.
    sub: 'A model fitted in your browser, right now, on tournament games from seasons '
       + 'before this one — never on the season you are looking at. It was picked by '
       + 'testing it against the alternatives, not by preference. Its equation is '
       + 'printed below, coefficient by coefficient — the only strategy here that shows its work.',
    tag: 'live',
    // Same two numbers as the other cards, from the same scorer, when the
    // evaluation on file is for exactly this bracket -- see fittedEval().
    // Otherwise the accuracy alone, which is what this card showed before an
    // evaluation existed at all.
    stat: (() => {
      const fe = fittedEval();
      if (fe && !fe.stale) return `${p1Pct(fe.p1)} to win · ${fe.ev.toFixed(0)} pts · evaluated`;
      return state.fit && state.fit.oos
        ? `${(state.fit.oos.accuracy * 100).toFixed(0)}% of games called right`
        : '';
    })(),
  });

  document.getElementById('strat-list').innerHTML = opts.map(o => `
    <label class="vopt${o.active || state.strategy === o.id ? ' active' : ''}">
      <input type="radio" name="strategy" ${(o.active || state.strategy === o.id) ? 'checked' : ''}
             onchange="setStrategy('${o.id}')">
      <span class="vopt-name">${o.label}</span>
      <span class="v-tag">${o.tag}</span>
      <span class="vopt-sub">${o.sub}</span>
      ${o.stat ? `<span class="vopt-stat">${o.stat}</span>` : ''}
    </label>`).join('');

  renderFilters();
  renderFilterNotes();
}

/* The help text under each filter row states what the numbers mean, so it has
 * to follow the objective. Under "expected points" the figures are points and
 * the cost of a longshot is points, not win probability -- describing them the
 * other way round was the same confusion the filters themselves used to have. */
function renderFilterNotes() {
  const inert = state.strategy === MODEL;
  const ev = state.objective === 'ev';
  const notes = ['champ-note', 'ones-note', 'shape-note', 'dd-note', 'src-note']
    .map(id => document.getElementById(id));
  const suffix = inert
    ? 'Filters apply to the two precomputed strategies; the fitted model builds its own board.'
    : ev
      ? 'Numbers are expected ESPN points. Backing a longer shot costs points — the number says how many.'
      : 'Numbers are the chance of finishing first in a 30-person pool. Backing a longer shot costs win probability — the number says how much.';
  // Plain words, and no repo paths. This list used to include "Bracket shapes
  // the pipeline already scores, from src/product/selection.py" -- a source
  // file path, shown to people who do not have the source.
  //
  // The 1-seed and depth rows also needed a correction underneath them
  // ("Depth says how far DOWN you reach; this says how much of the top you
  // keep"), which is a label admitting it failed. Retitled instead.
  const lead = [
    'Each is the best bracket available with that team winning it all.',
    'How many of the four 1 seeds you have reaching the Final Four.',
    'The lowest seed you have reaching the Final Four — how big an upset you are backing.',
    'Well-known bracket patterns, scored the same way as everything else here.',
    'Which rating system imagined this bracket. They disagree about real teams, which is the point.',
  ];
  const kinds = ['champ', 'ones', 'depth', 'pred', 'src'];
  notes.forEach((el, i) => {
    if (!el) return;
    const greyed = !inert && _unavailable[kinds[i]] > 0
      ? ' Greyed-out options have no bracket left that also matches your other filters.'
      : '';
    el.textContent = `${lead[i]} ${suffix}${greyed}`;
  });
}

/* Champion picker.
 *
 * WHY THIS EXISTS. The two objective strategies are much more alike than their
 * labels suggest -- in 2026 they agree on 55 of 63 games and share an identical
 * Final Four -- so a menu of two implied the model has one opinion. It does not:
 * the candidate pool carries a dozen viable champions by construction, and none
 * of them were reachable from the page.
 *
 * ORDERED BY P(1st), so the cost of backing an underdog is legible: the list
 * runs from the best available bracket down, and each figure is on the same
 * scale as the headline strategies. This is a menu of beliefs with prices
 * attached, not a shuffle button.
 */
/* The three filter rows.
 *
 * One renderer for all of them, because they differ only in which axis they set
 * and how a value is labelled. A chip is offered when some cell exists with that
 * value plus whatever else is currently selected, so availability is read from
 * the data rather than hard-coded, and the figure shown is the score of the
 * bracket that click would actually return.
 *
 * DIMMED, NOT HIDDEN. A chip that disappears when you select something else
 * reads as a bug; a dimmed one reads as a constraint.
 */
const FILTER_ROWS = [
  { kind: 'champ', host: 'champ-list', panel: 'champions' },
  { kind: 'ones', host: 'ones-list', panel: 'ones' },
  { kind: 'depth', host: 'shape-list', panel: 'shapes' },
  { kind: 'pred', host: 'dd-list', panel: 'dd16' },
  { kind: 'src', host: 'src-list', panel: 'sources' },
];

function renderFilters() {
  const s = state.season;
  const f = (s && s.filters) || null;
  const obj = state.objective;
  const inert = state.strategy === MODEL;

  for (const row of FILTER_ROWS) {
    const host = document.getElementById(row.host);
    const panel = document.getElementById(row.panel);
    if (!host || !panel) continue;
    if (!f) { panel.hidden = true; continue; }

    const values = row.kind === 'champ' ? f.champions.map(c => c.team)
                 : row.kind === 'ones' ? f.ones
                 : row.kind === 'depth' ? f.depths
                 : row.kind === 'pred' ? (f.predicates || []).map(x => x.i)
                 : f.sources;
    panel.hidden = !values || values.length === 0;
    if (panel.hidden) continue;

    let unavailable = 0;
    host.innerHTML = values.map(v => {
      const on = state.pick[row.kind] === v;
      // Availability is a query, not a table: does anything survive with this
      // value plus whatever else is selected.
      const rows = inert ? [] : matching({ ...state.pick, [row.kind]: v });
      const ok = rows.length > 0;
      if (!ok) unavailable++;
      const best = ok ? rows.reduce((a, b) => (b[obj] > a[obj] ? b : a)) : null;
      const stat = !best ? '—'
        : obj === 'ev' ? `${best.ev.toFixed(0)} pts` : p1Pct(best.p1);

      let lead = '', name = '';
      if (row.kind === 'champ') {
        const c = f.champions.find(x => x.team === v);
        lead = String(c.seed); name = c.name;
      } else if (row.kind === 'ones') {
        lead = String(v); name = v === 1 ? 'one-seed' : 'one-seeds';
      } else if (row.kind === 'depth') {
        lead = String(v); name = 'seed';
      } else if (row.kind === 'pred') {
        const pd = (f.predicates || []).find(x => x.i === v);
        lead = ''; name = pd ? pd.label : String(v);
      } else {
        lead = ''; name = SRC_LABEL[v] || v;
      }
      // How often this shape actually happens, from the simulated bank --
      // f.predicate_probabilities, which shipped in the payload and was read by
      // nothing. The candidate pool deliberately over-samples unlikely
      // champions for diversity, so counting matching rows is NOT a frequency
      // and the artifact says so in as many words. This is the honest number,
      // and it belongs on the chip rather than in a tooltip no phone can show.
      // Keyed by predicate NAME, while the chip's value is its index -- the
      // payload carries both on f.predicates, so go through the record.
      const predKey = row.kind === 'pred'
        ? ((f.predicates || []).find(x => x.i === v) || {}).key
        : undefined;
      const freq = predKey ? (f.predicate_probabilities || {})[predKey] : undefined;
      const freqText = freq === undefined ? '' : `${(freq * 100).toFixed(0)}% of simulated tournaments`;

      const title = inert ? 'Filters apply to the precomputed strategies, not the fitted model'
                  : !ok ? 'Nothing matches that with your other filters'
                  : freqText || 'Best bracket available with this choice';
      return `
        <button class="chip${on ? ' on' : ''}${ok ? '' : ' off'}" ${ok ? '' : 'disabled'}
                onclick="setFilter('${row.kind}', ${typeof v === 'string' ? `'${v}'` : v})" title="${title}">
          ${lead ? `<span class="chip-seed">${lead}</span>` : ''}
          <span class="chip-name">${name}</span>
          ${freqText ? `<span class="chip-freq">${freqText}</span>` : ''}
          <span class="chip-stat">${stat}</span>
        </button>`;
    }).join('');
    // Why some options are greyed out, in text rather than in a title= tooltip
    // that a phone will never show. Only said when it is actually true.
    _unavailable[row.kind] = unavailable;
  }

  renderAlts();
}

/* Near-tied alternates.
 *
 * The referee's standard error is about half a point, so the top few brackets in
 * any filtered set are not meaningfully ranked. Showing only the argmax presents
 * a coin flip as a verdict; this offers the tie and lets the user break it on
 * something the model cannot see. */
function renderAlts() {
  const host = document.getElementById('alt-list');
  const panel = document.getElementById('alts');
  if (!host || !panel) return;
  const { alts } = filteredEntry();
  const show = state.strategy === CUSTOM && alts && alts.length > 1;
  panel.hidden = !show;
  if (!show) { host.innerHTML = ''; return; }
  const obj = state.objective;
  // LABEL BY WHAT DIFFERS, NOT BY THE CHAMPION. Filtering to a champion makes
  // every alternate carry that same name, so a row of identical labels is no
  // help in choosing between them. The Final Four is the first place these
  // brackets actually diverge, so the label is whichever of its teams the
  // options disagree about.
  const f4Of = r => decodeBracket(r.b)[3];
  const common = f4Of(alts[0]).filter(x => alts.every(r => f4Of(r).includes(x)));
  host.innerHTML = alts.map((r, i) => {
    const on = Math.min(state.alt, alts.length - 1) === i;
    const stat = obj === 'ev' ? `${r.ev.toFixed(0)} pts` : p1Pct(r.p1);
    const t = state.season.teams;
    const distinct = f4Of(r).filter(x => !common.includes(x));
    const name = distinct.length
      ? distinct.map(x => t[x].name).join(' + ')
      : `${t[r.c].name} (${SRC_LABEL[r.s] || r.s})`;
    return `
      <button class="chip${on ? ' on' : ''}" onclick="setAlt(${i})"
              title="Final Four: ${f4Of(r).map(x => `${t[x].name} (${t[x].seed})`).join(', ')}">
        <span class="chip-seed">${i + 1}</span>
        <span class="chip-name">${name}</span>
        <span class="chip-stat">${stat}</span>
      </button>`;
  }).join('');
}

function setAlt(i) {
  state.alt = i;
  writeHash();
  renderStrategies();
  render();
}

/* ---------- getting the bracket out ----------
 *
 * The job this page exists to finish is 63 picks typed into a pool site. Until
 * now the only way to collect them was to read a six-column horizontally
 * scrolling board -- on a phone, one column at a time -- and retype it from
 * memory. Every modelling decision in this repo sits upstream of that step.
 *
 * Round-by-round winners, in bracket order, because that is the order the entry
 * form asks for them.
 */
/* Region-scoped rounds: the bracket only crosses regions from the Final Four
 * on, so a game in one of these four rounds always has both teams from the
 * same region -- structurally, by construction of a balanced bracket, not
 * something that happens to be true of one season's draw (see the review
 * note this fixes, below). Fixed at 4 for the same reason the rest of this
 * file hardcodes a 6-round, 4-region, 64-team bracket rather than deriving
 * it: this codebase does not model any other size. */
const REGION_SCOPED_ROUNDS = 4;

/* 2026-09 site review: this export named a round's WINNERS under that round's
 * label, which reads as a mismatch either way you take it -- "Round of 64"
 * headed the 32 teams who won their way OUT of it, and "Final Four" headed
 * only the 2 teams who beat the other two. Worse, a winner name alone does
 * not say which game it belongs to, which is the one thing an ESPN entry
 * form actually asks: not "who survived", but "who won THIS matchup".
 *
 * The fix is not a label shift -- ROUNDS[r] already names the round game g
 * was played in correctly, because state.rounds[r] holds that round's real
 * games (g.a, g.b, g.win), the same shape the board itself renders. Printing
 * both sides of every game, not just the winner, makes the header and the
 * body describe the same round and gives every line something to match
 * against the entry form. Grouping by region for the four rounds where that
 * is a real property of the game (not the Final Four or the Championship,
 * which cross regions by definition) turns a flat 32-line dump into
 * something organised the way ESPN's own bracket is.
 */
function picksAsText() {
  if (!state.rounds || !state.season) return '';
  const st = usingOptimized() ? currentStrategy() : null;
  // g.win is an INDEX into season.teams, the same thing sideHTML() resolves.
  // Treating it as a team id silently produced "undefined 8" for every pick.
  const team = i => state.season.teams[i] || {};

  const head = [
    `${state.year} bracket — ${st ? st.label : 'Fitted model'}`,
    st && st.p1 !== undefined
      ? `${p1Pct(st.p1)} to finish first, ${st.ev.toFixed(0)} expected points`
      : '',
    st && st.id === RULE ? `Experimental rule search — not scored against the pool. ${st.note}` : '',
    // The fitted bracket's numbers travel too, labelled for what they are.
    (() => { const fe = !st ? fittedEval() : null;
             return fe && !fe.stale
               ? `${p1Pct(fe.p1)} to finish first, ${fe.ev.toFixed(0)} expected points ` +
                 `(evaluated in the common pool framework; not a selected candidate)`
               : ''; })(),
    // The disclosure travels with the picks. A bracket pasted into a group chat
    // outlives the page it came from, and the number goes with it.
    state.season.p1_assumption || '',
  ].filter(Boolean);

  const gameLine = g => {
    const a = team(g.a), b = team(g.b);
    const won = g.win === g.a, w = won ? a : b, l = won ? b : a;
    return `${w.seed} ${w.name} over ${l.seed} ${l.name}`;
  };

  const body = state.rounds.map((games, r) => {
    const lines = [];
    let lastRegion;   // undefined until the first game sets it, deliberately
    for (const g of games) {
      const region = r < REGION_SCOPED_ROUNDS ? team(g.a).region : null;
      if (region && region !== lastRegion) { lines.push(`  ${region}`); lastRegion = region; }
      lines.push(`  ${gameLine(g)}`);
    }
    return `${ROUNDS[r]}\n${lines.join('\n')}`;
  });

  return `${head.join('\n')}\n\n${body.join('\n\n')}\n`;
}

function copyPicks() {
  const text = picksAsText();
  const msg = document.getElementById('copy-msg');
  const say = t => { if (msg) { msg.textContent = t; setTimeout(() => { msg.textContent = ''; }, 4000); } };
  if (!text) return say('Nothing to copy yet.');

  // navigator.clipboard needs a secure context. The published site is https, but
  // a local file:// or plain-http preview is not, and failing silently there
  // would make this look broken exactly where it gets tested.
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    document.body.removeChild(ta);
    say(ok ? 'Picks copied.' : 'Copy failed — use Print instead.');
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => say('Picks copied.'), fallback);
  } else {
    fallback();
  }
}

function setFilter(kind, value) {
  state.alt = 0;
  // Inert under the fitted model: that board is derived live from a regression,
  // so there is no candidate pool to narrow.
  if (state.strategy === MODEL) return;
  const prev = { ...state.pick };
  state.pick[kind] = (state.pick[kind] === value) ? null : value;

  // A combination the pool cannot fill would blank the board. Rather than
  // refuse the click, drop the OTHER axes in the order they were least recently
  // meaningful -- the user's newest intent is the one to honour.
  // A combination nothing satisfies would blank the board. Drop other axes
  // rather than refuse the click -- the newest intent is the one to honour.
  const dropped = [];
  if (!matching().length) {
    for (const other of ['pred', 'src', 'depth', 'ones', 'champ']) {
      if (other === kind || state.pick[other] === null) continue;
      state.pick[other] = null;
      dropped.push(other);
      if (matching().length) break;
    }
  }
  let clearedAll = false;
  if (!matching().length) {
    state.pick = { champ: null, ones: null, depth: null, pred: null, src: null };
    state.pick[kind] = value;
    clearedAll = true;
  }
  if (!matching().length) { state.pick = prev; clearedAll = false; dropped.length = 0; }

  // Dropping the other filters is deliberate -- the newest click is the intent
  // to honour -- but it used to happen in silence, with chips simply going
  // dark. A control that undoes your previous choices without saying so reads
  // as broken rather than as helpful.
  const label = { champ: 'champion', ones: 'one-seeds', depth: 'Final Four depth', pred: 'bracket shape', src: 'model' };
  state.notice = clearedAll
    ? 'No bracket matched all of those, so the other filters were cleared to honour this one.'
    : dropped.length
      ? `No bracket matched that with your ${dropped.map(k => label[k]).join(' and ')} filter, so it was dropped.`
      : '';

  state.strategy = anyFilter() ? CUSTOM : state.objective;
  writeHash();
  refit();
  renderStrategies();
  render();
}

function setStrategy(id) {
  // Choosing an objective KEEPS the filters and re-resolves them under the new
  // question, which is the whole point of separating the two. Only the fitted
  // model clears them, because it has no pool to filter.
  if (id === 'p1' || id === 'ev') {
    state.objective = id;
    state.strategy = anyFilter() ? CUSTOM : id;
  } else {
    if (id === MODEL || id === RULE) state.pick = { champ: null, ones: null, depth: null, pred: null, src: null };
    state.strategy = id;
  }
  state.notice = '';
  refit();
  writeHash();
  renderStrategies();
  render();
  if (id === RULE) ensureRuleSearch();
}

/* ---------- team drawer ---------- */

/* Each round's chance of reaching it, for the drawer.
 *
 * Reuses state.advancement -- the same recursive, real-bracket calculation
 * the board's per-game percentages come from (see refit(), fitReady(),
 * bracketAdvancementProbs() in fit.js) -- rather than approximating it from
 * this team's own game-by-game percentages, which would silently ignore
 * whether each future opponent is themselves likely to arrive.
 *
 * ROUNDS[0], the Round of 64, is the fixed starting field rather than a
 * predicted outcome, so it has nothing to show here; probs[r-1] is "reached
 * ROUNDS[r]" for r >= 1, matching the numbering solveByFit() already uses.
 */
function advancementHTML(i) {
  const probs = state.advancement && state.advancement[i];
  if (!probs) return '';
  return `
    <div class="d-group">
      <p class="g-name">Model's chance of reaching each round</p>
      ${ROUNDS.slice(1).map((label, ri) => {
        const p = probs[ri];
        const pct = Math.max(2, Math.min(98, p * 100));
        return `
        <div class="d-row">
          <span class="d-lab">${label}</span>
          <span class="d-track"><i style="left:${pct}%"></i></span>
          <span class="d-val">${p1Pct(p)}</span>
        </div>`;
      }).join('')}
    </div>`;
}

/* Where this team's value sits among the 64 in the field, 0-100, in the
 * BETTER direction -- or null if the value is missing.
 *
 * Uses the payload's z column rather than raw, because z is already
 * sign-corrected in build_ui_payload.py (higher_better=False stats are
 * negated before standardising), so "larger z" means "better" for every
 * variable and one comparison serves all of them. Missing values are
 * excluded by checking RAW, not z: a missing raw is shipped as z = 0, which
 * would otherwise count as a perfectly average team and be ranked.
 *
 * Ties get half credit (the usual percentile-rank convention), so two teams
 * with identical values get the same number rather than an arbitrary order.
 *
 * WHY A RANK AND NOT THE OLD DOT. The drawer used to place its marker at
 * 50 + 16 z, a linear map of z that read as a percentile and was not one:
 * z is standardised over the whole ~360-team D1 field, so most of the 64
 * here sit far to the right of centre and the dot said little about how a
 * team compared to the field it actually has to beat. And a raw number
 * alone -- "Defense 91.0", "Ball security 0.158" -- says nothing about
 * direction at all (2026-09 site review).
 */
function percentileInField(zs, raws, i) {
  if (!zs || !raws) return null;
  const mine = raws[i];
  if (mine === null || mine === undefined) return null;
  const z = zs[i];
  let below = 0, ties = 0, n = 0;
  for (let j = 0; j < raws.length; j++) {
    if (raws[j] === null || raws[j] === undefined) continue;
    n++;
    if (zs[j] < z) below++;
    else if (zs[j] === z) ties++;   // includes j === i
  }
  if (!n) return null;
  return 100 * (below + 0.5 * (ties - 1)) / Math.max(n - 1, 1);
}

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function openTeam(i) {
  const s = state.season, t = s.teams[i];
  document.getElementById('d-name').textContent = t.name;
  document.getElementById('d-sub').textContent = `${t.seed} seed \u00b7 ${t.region}`;

  const groups = {};
  for (const v of s.variables) (groups[v.group] ||= []).push(v);

  // Said once, up top, rather than in per-row title= tooltips a phone will
  // never show: which way is "good" for each raw number, and what the
  // percentile means. Both used to be unstated, and the raw numbers for
  // lower-is-better stats read backwards without them.
  const legend = `
    <p class="d-legend">
      Percentile is this team's rank among the ${s.teams.length} in the field, always
      in the better direction. <span class="d-dir">\u2193</span> marks stats where a
      lower raw number is better; the percentile already accounts for that.
    </p>`;

  document.getElementById('d-body').innerHTML = advancementHTML(i) + legend + Object.entries(groups).map(([g, vars]) => `
    <div class="d-group">
      <p class="g-name">${g}</p>
      ${vars.map(v => {
        const raw = (s.raw[v.key] || [])[i];
        const missing = raw === null || raw === undefined;
        const pct = missing ? null : percentileInField(s.z[v.key], s.raw[v.key], i);
        // Lit means "this one is in the fitted model", which is now a fact to
        // read rather than a control to operate. Every stat is still shown,
        // because the drawer is for understanding a team, not for configuring
        // a model.
        const on = CANONICAL_KEYS.indexOf(v.key) >= 0;
        return `
        <div class="d-row${on ? ' lit' : ''}">
          <span class="d-lab">${v.label}${v.higher_better ? '' : ' <span class="d-dir">\u2193</span>'}</span>
          <span class="d-track">${pct === null ? '' : `<i style="left:${Math.max(2, Math.min(98, pct))}%"></i>`}</span>
          <span class="d-val">${missing ? '\u2014' : fmt(raw)}</span>
          <span class="d-pct">${pct === null ? '' : ordinal(Math.round(pct))}</span>
        </div>`;
      }).join('')}
    </div>`).join('');

  document.getElementById('drawer').hidden = false;
  document.getElementById('scrim').hidden = false;
}

function fmt(v) {
  if (Number.isInteger(v)) return String(v);      // ranks and counts
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}

function closeDrawer() {
  document.getElementById('drawer').hidden = true;
  document.getElementById('scrim').hidden = true;
}

/* ---------- controls ---------- */

async function setYear(year) {
  // A team drawer is keyed by INDEX into state.season.teams, a slot that is
  // only stable within one season's bracket -- index 5 in 2011 and index 5 in
  // 2026 are unrelated teams. Leaving the drawer open across a year change
  // used to keep showing the old season's name, stats, and (since this
  // change added round-advancement numbers computed from the new season's
  // bracket) an increasingly incoherent mix of old and new data for a team
  // that may not even be the one on screen. There is no correct team to
  // refresh it to, so closing it is the only choice that cannot show
  // something wrong.
  closeDrawer();
  state.year = year;
  state.notice = '';
  // A new season is a new bracket top to bottom; the mobile board goes back
  // to its default round rather than wherever the old season left it.
  state.mobileRound = MOBILE_ROUND_DEFAULT;
  document.querySelectorAll('.yr').forEach(b => b.classList.toggle('on', Number(b.dataset.year) === year));
  try {
    state.season = await loadSeason(year);
  } catch {
    state.season = null;
  }
  reconcileFiltersWithSeason();
  writeHash();
  // Refit: the excluded season changed, so the coefficients must change too.
  refit();
  renderStrategies();
  render();
  // The rule search is keyed on the season too: a held result is for the
  // previous one and ruleStrategy() will not show it, so the search reruns
  // here -- this is also how a link opening on s=rule gets its first result.
  if (state.strategy === RULE) ensureRuleSearch();
}

async function init() {
  const [idx] = await Promise.all([
    fetch(`data/seasons.json?v=${DATA_V}`).then(r => r.json()),
    loadTraining(),
  ]);
  // Open on the newest season that actually has a bracket. Not max(year):
  // 2027 is listed from the moment the calendar knows about it and stays
  // "not_started" until Selection Sunday, so the newest LISTED season is an
  // empty state for most of the year.
  // A shared link names its own season; otherwise open on the newest ready one.
  state.seasonsIndex = idx.seasons;
  const fromHash = readHash();
  const fallback = pickDefaultSeason(idx.seasons);
  if (fromHash !== null && idx.seasons.some(s => s.year === fromHash)) {
    state.year = fromHash;
  } else if (fallback !== null) {
    state.year = fallback;
  }

  document.getElementById('years').innerHTML = idx.seasons.map(s => `
    <button class="yr${s.year === state.year ? ' on' : ''}${s.status === 'ready' ? '' : ' na'}"
            data-year="${s.year}" title="${s.status === 'ready' ? '' : `No bracket available for ${s.year}`}"
            onclick="setYear(${s.year})">${s.year}</button>`).join('');

  document.getElementById('d-close').addEventListener('click', closeDrawer);
  document.getElementById('scrim').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
  wireBoardSwipe();

  // Pasting a link into the address bar of the page you are already on is a
  // same-document navigation: nothing reloads and, without this, nothing
  // happens. writeHash() uses replaceState, which does NOT fire hashchange, so
  // this cannot loop on our own writes.
  window.addEventListener('hashchange', () => location.reload());

  await setYear(state.year);

  // The row scrolls on a narrow viewport and the default season sits at the
  // far right of seventeen, so it would otherwise open out of view.
  document.querySelector('.yr.on')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

init();
