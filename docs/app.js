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

/* The browser-fitted strategy. Anything else is a precomputed bracket read out
 * of the season payload by id. */
const MODEL = 'model';
const RULE = 'rule';
const RULE_CHECKPOINTS = [{ r: 1, label: 'Sweet 16' }, { r: 2, label: 'Elite Eight' }, { r: 3, label: 'Final Four' }, { r: 5, label: 'Champion' }];

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
  strategy: 'p1',       // 'p1' | 'ev' | MODEL
  objective: 'p1',
  risk: 0.35,
  pick: { champ: null, ones: null, depth: null, pred: null, src: null },
  notice: '',
  explore: 'adj_defensive_efficiency',
  exploreTab: 'field',
  mobileRound: 0,
  boardDetail: '',
  rule: { checkpoints: [3, 5], maxCriteria: 2, result: null, busy: false, error: null, chosen: 0 },
  fit: null,            // {beta, n, converged}
  advancement: null,    // {team: [P(reach R32), ..., P(win it all)]}, see refit()
  // Strategy-card family filter, added with the 2026-09-19 card restyle
  // (docs/index2.html). Purely a display filter over strategyRows()'s own
  // `kind` string -- there are only 3 real strategies, so this narrows
  // which cards show rather than changing what any of them compute.
  // 'all' | 'backtested' | 'optimal' | 'fitted'
  family: 'all',
  seasonsIndex: null,   // seasons.json, kept so pickDefaultSeason() knows which seasons were played
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

/* Candidate-pool controls retained from the original UI. */
const CUSTOM = 'custom';
const AXIS_FIELD = { champ: 'c', ones: 'o', depth: 'd', src: 's' };
const SRC_LABEL = { torvik: 'Torvik', massey_avg: 'Massey', elo: 'Elo', region_top_n: 'Region construction', shipped: 'Recommended' };
function candidates() { return ((state.season && state.season.filters) || {}).candidates || []; }
function matching(pick = state.pick) {
  return candidates().filter(r => Object.entries(AXIS_FIELD).every(([k, f]) => pick[k] == null || r[f] === pick[k]) && (pick.pred == null || r.k[pick.pred] === '1'));
}
function anyFilter() { return Object.values(state.pick).some(v => v !== null); }
function decodeBracket(bits) {
  let cur = state.season.first_round.slice(), i = 0; const rounds = [];
  for (let r = 0; r < 6; r++) { const next = []; for (let g = 0; g < cur.length; g += 2) next.push(bits[i++] === '1' ? cur[g] : cur[g + 1]); rounds.push(next); cur = next; }
  return rounds;
}
function filteredEntry() {
  if (!anyFilter()) return { entry: null, alts: [] };
  const rows = matching(); if (!rows.length) return { entry: null, alts: [] };
  const by = { p1: rows.reduce((a, b) => b.p1 > a.p1 ? b : a), ev: rows.reduce((a, b) => b.ev > a.ev ? b : a) };
  return { entry: { n: rows.length, by, row: by[state.objective] }, alts: rows.slice().sort((a, b) => b[state.objective] - a[state.objective]).slice(0, 5) };
}
function riskCandidate() {
  if (!state.season || !state.season.filters) return null;
  const rows = candidates().filter(r => r.s === 'region_top_n' || r.s === 'shipped');
  if (!rows.length) return null;
  const maxUpsets = Math.max(...rows.map(r => r.o || 0), 1);
  const target = state.risk * maxUpsets;
  return rows.slice().sort((a, b) => Math.abs((a.o || 0) - target) - Math.abs((b.o || 0) - target) || b.p1 - a.p1)[0];
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
  const all = (seasons || []).slice().sort((a, b) => a.year - b.year);
  const ready = all.filter(s => s.status === 'ready').map(s => s.year);
  // The newest listed season is the front door once it can show something:
  // its bracket, or -- from the day the previous season is played until
  // Selection Sunday -- the rule search over the seasons before it, with a
  // blank bracket waiting for the field (see fieldPending()).
  const newest = all[all.length - 1];
  if (newest && newest.status === 'not_started' && ready.some(y => y < newest.year)) return newest.year;
  if (ready.length) return Math.max(...ready);
  // No bracket anywhere: show the newest thing we know about and let its own
  // empty state explain itself, rather than a year that may not be listed.
  return all.length ? all[all.length - 1].year : null;
}

/* Whether the live fitted model (fit.js) can score an arbitrary matchup right
 * now, regardless of which strategy is on screen.
 *
 * The precomputed strategies carry a whole-bracket score rather than per-game
 * probabilities, so this is the source for the fitted card alone. */
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

/* ---------- render ---------- */

function render() {
  const s = state.season;
  { const main = document.getElementById('main'); if (main) main.classList.toggle('pending', fieldPending(s)); }
  renderCountdown();
  renderYears();
  const unavail = document.getElementById('unavailable');
  // Should not happen once at least one season is ready, but a season
  // payload can fail to load (loadSeason() in setYear()) -- said plainly
  // rather than left as an empty grid with no explanation.
  const broken = !fieldPending(s) && (!s || s.status !== 'ready');
  if (unavail) {
    unavail.hidden = !broken;
    if (broken) unavail.textContent = s ? (s.message || 'Season unavailable.') : 'Season unavailable.';
  }
  renderCompare();
  renderAdjust();
  renderBoard();
  renderRulePanel();
  renderExplore();
  renderLeaderboard();
}

function writeHash() {
  const p = new URLSearchParams({ y: String(state.year), s: state.strategy, o: state.objective, risk: String(state.risk) });
  Object.entries(state.pick).forEach(([k, v]) => { if (v !== null) p.set(k, String(v)); });
  p.set('rc', state.rule.checkpoints.join(',')); p.set('rm', String(state.rule.maxCriteria));
  history.replaceState(null, '', `#${p}`);
}
function readHash() {
  const p = new URLSearchParams((location.hash || '').slice(1));
  const y = Number(p.get('y')); if (Number.isFinite(y) && y > 1900) state.year = y;
  if (['p1', 'ev'].includes(p.get('o'))) state.objective = p.get('o');
  if (p.has('risk')) state.risk = Math.max(0, Math.min(1, Number(p.get('risk')) || .35));
  if ([MODEL, RULE, 'p1', 'ev'].includes(p.get('s'))) state.strategy = p.get('s');
  for (const k of Object.keys(state.pick)) { if (p.has(k)) state.pick[k] = k === 'src' ? p.get(k) : Number(p.get(k)); }
  if (p.has('rc')) state.rule.checkpoints = p.get('rc').split(',').map(Number).filter(Number.isFinite);
  if (p.get('rm') === '3') state.rule.maxCriteria = 3;
}
function renderYears() {
  const nav = document.getElementById('years'); if (!nav || !state.seasonsIndex) return;
  nav.innerHTML = state.seasonsIndex.filter(s => s.status === 'ready').sort((a,b) => b.year-a.year).map(s => `<button class="year-btn${s.year === state.year ? ' on' : ''}" onclick="setYear(${s.year})">${s.year}</button>`).join('');
}
function resetView() { state.strategy = 'p1'; state.objective = 'p1'; state.pick = { champ: null, ones: null, depth: null, pred: null, src: null }; state.rule = { checkpoints: [3, 5], maxCriteria: 2, result: null, busy: false, error: null, chosen: 0 }; state.exploreTab = 'field'; history.replaceState(null, '', location.pathname); render(); }

function renderBoard() {
  const root = document.getElementById('bracket-root');
  const pending = document.getElementById('bracket-pending');
  const title = document.getElementById('bracket-title');
  if (!root) return;
  const s = state.season;
  if (!s || s.status !== 'ready' || !Array.isArray(s.first_round)) {
    if (pending) pending.hidden = false;
    if (title) title.textContent = `${state.year || ''} Bracket`;
    return;
  }
  if (pending) pending.hidden = true;
  if (title) title.textContent = `${state.year} Bracket · ${state.strategy === MODEL ? 'Fitted model' : state.strategy === RULE ? 'Rule search' : state.objective === 'ev' ? 'Expected points' : 'P(1st)'}`;
  let rounds = null;
  if (state.strategy === MODEL && fitReady()) rounds = solveByFit();
  else if (state.strategy === RULE && state.rule.result && state.rule.result.entries.length) rounds = state.rule.result.entries[state.rule.chosen].rounds;
  else {
    const st = state.strategy === CUSTOM ? filteredEntry().entry?.row : (state.strategy === 'p1' && !anyFilter() ? riskCandidate() : (s.strategies || []).find(x => x.id === state.strategy));
    if (st) {
      const picks = st.b ? decodeBracket(st.b) : st.picks;
      let field = s.first_round.slice(); rounds = picks.map(winners => {
        const games = []; for (let i = 0; i < field.length; i += 2) games.push({ a: field[i], b: field[i + 1], win: winners[i / 2] });
        field = winners; return games;
      });
    }
  }
  if (!rounds) return;
  const names = i => s.teams[i] || { name: 'TBD', seed: '—' };
  root.innerHTML = rounds.map((games, r) => `<div class="region-col board-round${r === state.mobileRound ? ' mobile-active' : ''}"><div class="round-label">${['Round of 64','Round of 32','Sweet 16','Elite Eight','Final Four','Championship'][r]}</div>${games.map(g => { const a = names(g.a), b = names(g.b); return `<div class="matchup" onclick="showMatchup(${g.a},${g.b},${g.win},${g.p == null ? 'null' : g.p})"><div class="team-slot${g.win === g.a ? ' selected' : ''}"><span><span class="seed">${a.seed}</span>${a.name}</span>${g.win === g.a ? '<span>✓</span>' : ''}</div><div class="team-slot${g.win === g.b ? ' selected' : ''}"><span><span class="seed">${b.seed}</span>${b.name}</span>${g.win === g.b ? '<span>✓</span>' : ''}</div></div>`; }).join('')}</div>`).join('');
  const detail = document.getElementById('board-detail'); if (detail) detail.textContent = state.boardDetail;
}

function setMobileRound(delta) { state.mobileRound = Math.max(0, Math.min(5, state.mobileRound + delta)); renderBoard(); }
function showMatchup(a, b, win, p) { const s = state.season; if (!s || !s.teams[a] || !s.teams[b]) return; const ta = s.teams[a], tb = s.teams[b], winner = s.teams[win]; state.boardDetail = `${ta.seed} ${ta.name} vs ${tb.seed} ${tb.name} · pick: ${winner.name}${p == null ? '' : ` · fitted probability ${Math.round((win === a ? p : 1 - p) * 100)}%`}`; renderBoard(); }
async function copyPicks() {
  const s = state.season; if (!s || !s.first_round) return;
  let rounds = state.strategy === MODEL ? solveByFit() : (state.strategy === RULE && state.rule.result?.entries.length ? state.rule.result.entries[state.rule.chosen].rounds : null);
  if (!rounds) {
    const st = state.strategy === CUSTOM ? filteredEntry().entry?.row : (s.strategies || []).find(x => x.id === state.strategy);
    if (st) { const picks = st.b ? decodeBracket(st.b) : st.picks; let field = s.first_round.slice(); rounds = picks.map(winners => { const games = []; for (let i = 0; i < field.length; i += 2) games.push({ a: field[i], b: field[i + 1], win: winners[i / 2] }); field = winners; return games; }); }
  }
  const text = rounds ? rounds.flat().map(g => s.teams[g.win].name).join('\n') : 'Bracket picks are not available yet.';
  try { await navigator.clipboard.writeText(text); } catch { window.prompt('Copy these picks:', text); }
}

/* ---------- headline and comparison ----------
 *
 * The page used to open on three explanatory cards, a disclosure, the filter
 * stack, a strategy note, an equation and only then the bracket: a research
 * dashboard that happened to emit a bracket. The methodology audit
 * (artifacts/methodology_audit/) made the numbers on this page much stronger
 * and changed almost nothing a visitor could see, which was the right
 * outcome for an audit and the wrong state for a product. The structure is
 * now: choose a strategy -> compare the cards -> see the current-season
 * placeholder bracket. Nothing here computes anything new: the cards read
 * directly from the season payload.
 */

/* One row per selectable bracket, from the same fields the cards use.
 *
 * The fitted row's champion is always the live fit's; its P(1st)/EV exist only
 * when the evaluation on file is for exactly that bracket (fittedEval()). */
function strategyRows() {
  const s = state.season;
  if (fieldPending(s)) {
    // Listed for what they will be, once the field is out.
    const wait = `Awaiting the ${s.year} field`;
    return [
      { id: 'p1', label: 'Win the pool', kind: `Backtested rule · ${wait}`, p1: null, ev: null, champion: null, record: null, active: false, pending: true },
      { id: 'ev', label: 'Most expected points', kind: `Exact optimum · ${wait}`, p1: null, ev: null, champion: null, record: null, active: false, pending: true },
      { id: MODEL, label: 'Fitted model', kind: `Evaluated, not selected · ${wait}`, p1: null, ev: null, champion: null, record: null, active: false, pending: true },
      { id: RULE, label: 'Rule search', kind: `Experimental · ${wait}`, p1: null, ev: null, champion: null, record: null, active: false, pending: true },
    ];
  }
  // No listed season has a usable bracket at all (should not happen once one
  // season is ready, but a season payload can fail to load -- see setYear()).
  if (!s || s.status !== 'ready') return [];
  const teams = s.teams;
  const filt = state.strategy === MODEL ? null : filteredEntry().entry;
  const rows = (s.strategies || []).map(st => {
    const risk = st.id === 'p1' && !anyFilter() ? riskCandidate() : null;
    const value = risk || (filt && filt.by[st.id]) || st;
    const picks = risk || (filt && filt.by[st.id]) ? decodeBracket(value.b) : st.picks;
    return {
      id: st.id,
      label: st.id === 'ev' ? 'Most expected points' : 'Win the pool',
      kind: st.id === 'ev' ? 'Exact optimum' : 'Backtested rule',
      p1: value.p1, ev: value.ev, champion: teams[picks[5][0]], filtered: !!filt || !!risk,
      record: trackRecord(st.id),
      active: state.strategy === st.id || (state.strategy === CUSTOM && state.objective === st.id),
    };
  });
  const rr = state.rule.result;
  rows.push({ id: RULE, label: 'Rule search', kind: 'Experimental rule search', p1: null, ev: null,
    champion: rr && rr.entries && rr.entries[0] && rr.entries[0].picks ? teams[rr.entries[0].picks[5][0]] : null,
    record: null, active: state.strategy === RULE, pending: false });
  const fe = fittedEval();
  const live = fitReady() ? solveByFit() : null;
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

/* Which family chip a row falls under. Derived from the same `kind` string
 * the card's tag prints, so a chip and a card can never name a row
 * differently -- there is exactly one definition of what a strategy is. */
function strategyFamily(kind) {
  if (kind.startsWith('Backtested rule')) return 'backtested';
  if (kind.startsWith('Exact optimum')) return 'optimal';
  return 'fitted';
}
const FAMILY_LABEL = { all: 'All', backtested: 'Backtested rule', optimal: 'Exact optimum', fitted: 'Fitted model' };

/* One line under each card's name -- the only explanation of what a strategy
 * is left on the page since the methodology panel was removed 2026-09-19
 * (see docs/index2.html). */
function scardNote(id) {
  if (id === 'p1') return 'Fixed contrarian-risk rule, backtested across played seasons: takes upsets the field will not because second place pays nothing.';
  if (id === 'ev') return 'The exact expected-points maximum over the candidate pool, solved by dynamic programming.';
  if (id === RULE) return 'Experimental one-variable-per-round rules searched across prior tournaments. Not scored as a pool strategy.';
  return 'A ridge regression fitted in your browser on seasons before this one, never on the one shown.';
}

function setFamily(val) {
  state.family = val;
  renderCompare();
}

function renderCompare() {
  const box = document.getElementById('compare');
  if (!box) return;
  const rows = strategyRows().map(r => ({ ...r, family: strategyFamily(r.kind) }));
  const cell = (v, f) => (v === null || v === undefined ? '<span class="muted">—</span>' : f(v));
  const families = ['all', ...Array.from(new Set(rows.map(r => r.family)))];
  const shown = state.family === 'all' ? rows : rows.filter(r => r.family === state.family);
  box.hidden = false;
  box.innerHTML = `
    <div class="objective-toggle"><span>Optimise for</span><button class="chip${state.objective === 'p1' ? ' on' : ''}" onclick="setObjective('p1')">Maximise P(1st)</button><button class="chip${state.objective === 'ev' ? ' on' : ''}" onclick="setObjective('ev')">Maximise expected points</button></div>
    ${families.length > 2 ? `<div class="family-chips">${families.map(f => `
      <button class="chip${state.family === f ? ' on' : ''}" onclick="setFamily('${f}')">${FAMILY_LABEL[f] || f}</button>`).join('')}</div>` : ''}
    <div class="strategy-grid">${shown.map(r => `
      <div class="scard${r.pending && r.id !== RULE ? ' na' : r.active ? ' on' : ''}"
        ${r.pending && r.id !== RULE ? 'aria-disabled="true"' : `onclick="setStrategy('${r.id}')" role="button" tabindex="0"
          onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setStrategy('${r.id}'); }"`}>
        <span class="scard-tag">${r.kind}${r.filtered ? ' · filtered' : ''}</span>
        <h4>${r.label}${r.id === 'p1' ? ' <span class="recommended">Recommended starting point</span>' : ''}</h4>
        <p class="scard-note${r.stale ? ' stale' : ''}">${r.stale ? 'Not scored: the evaluation on file is for a different bracket.' : scardNote(r.id)}</p>
        <div class="scard-row"><span>Objective</span><b>${r.id === 'p1' ? 'Win the pool' : r.id === 'ev' ? 'Expected points' : r.id === MODEL ? 'Game prediction' : 'Historical reproduction'}</b></div>
        <div class="scard-row"><span>Chance of 1st</span><b>${cell(r.p1, p1Pct)}</b></div>
        <div class="scard-row"><span>Exp. points</span><b>${cell(r.ev, v => v.toFixed(0))}</b></div>
        <div class="scard-row"><span>Champion</span><b>${r.champion ? `${r.champion.seed} ${r.champion.name}` : '—'}</b></div>
        ${r.record ? `<div class="scard-row real"><span>Scored</span><b>${r.record.points.toLocaleString()}</b></div>
        <div class="scard-row real"><span>Finish</span><span>${finishText(r.record)}</span></div>` : ''}
        ${r.id === 'p1' && !r.pending ? `<div class="risk-gauge">
          <div class="risk-track"><span class="risk-mark" style="left:35%"></span></div>
          <div class="risk-label"><span>Chalk</span><span>Chaos</span></div>
        </div>` : ''}
      </div>`).join('')}
    </div>
    <table class="comparison"><thead><tr><th>Strategy</th><th>Type</th><th>P(1st)</th><th>Expected points</th><th>Champion</th></tr></thead><tbody>${shown.map(r => `<tr><td>${r.label}</td><td>${r.kind}</td><td>${cell(r.p1, p1Pct)}</td><td>${cell(r.ev, v => v.toFixed(0))}</td><td>${r.champion ? `${r.champion.seed} ${r.champion.name}` : '—'}</td></tr>`).join('')}</tbody></table>
    <p class="cmp-foot">P(1st) is the expected share of first place in the modeled 30-entry pool, not a universal probability of winning any pool.</p>
    ${state.notice ? `<p class="cmp-foot">${state.notice}</p>` : ''}
    ${rows.some(r => r.record) ? `<p class="cmp-foot">Chance and expected points are what the model expected before the tournament;
      Scored and Finish are what happened, against the same simulated 30-entry fields the chance was measured in.
      One season is one draw.</p>` : ''}`;
}

function renderAdjust() {
  const box = document.getElementById('adjust'); if (!box) return;
  const f = state.season && state.season.filters;
  if (!f || state.strategy === MODEL || fieldPending()) { box.hidden = true; return; }
  box.hidden = false;
  const rows = [
    ['champ', 'Champion', (f.champions || []).map(x => [x.team, `${x.seed} ${x.name}`])],
    ['ones', '1-seeds in Final Four', (f.ones || []).map(x => [x, String(x)])],
    ['depth', 'Lowest Final Four seed', (f.depths || []).map(x => [x, `${x}-seed`])],
    ['pred', 'Bracket shape', (f.predicates || []).map(x => [x.i, x.label])],
    ['src', 'Model source', (f.sources || []).map(x => [x, SRC_LABEL[x] || x])],
  ];
  const matches = matching();
  const active = Object.entries(state.pick).filter(([, v]) => v !== null).map(([kind, value]) => `<button class="chip on" onclick="setFilter('${kind}',${JSON.stringify(value)})">${kind}: ${kind === 'src' ? (SRC_LABEL[value] || value) : value} ×</button>`).join('');
  const riskControl = state.strategy === 'p1' || state.strategy === CUSTOM ? `<div class="risk-control"><label for="risk-slider"><b>Contrarian risk</b> <span>${state.risk < .25 ? 'Chalk-heavy' : state.risk > .65 ? 'Upset-heavy' : 'Balanced'}</span></label><input id="risk-slider" type="range" min="0" max="100" value="${Math.round(state.risk * 100)}" oninput="setRisk(this.value)"><div class="risk-scale"><span>Chalk</span><span>Validated default</span><span>Chaos</span></div><p class="adjust-help">Controls the P(1st) construction only. Expected points, fitted, and rule-search strategies do not optimize this parameter.</p></div>` : `<p class="adjust-help">Risk control applies only to the P(1st) construction; this strategy has its own fixed objective.</p>`;
  box.innerHTML = `<h3>Adjust this bracket</h3><p class="adjust-help">These choices narrow the candidate pool; the selected objective still decides what is maximized.</p>${riskControl}<div class="active-filters">${active || '<span class="muted">No filters applied</span>'}</div><p class="result-count">${matches.length.toLocaleString()} candidate bracket${matches.length === 1 ? '' : 's'} match</p>
    <div class="objective-row"><button class="chip${state.objective === 'p1' ? ' on' : ''}" onclick="setObjective('p1')">Maximise P(1st)</button><button class="chip${state.objective === 'ev' ? ' on' : ''}" onclick="setObjective('ev')">Maximise expected points</button></div>
    ${rows.map(([kind, label, values]) => `<div class="filter-row"><b>${label}</b><div class="chip-list">${values.map(([v, name]) => { const ok = matching({ ...state.pick, [kind]: v }).length > 0; return `<button class="chip${state.pick[kind] === v ? ' on' : ''}${ok ? '' : ' off'}" ${ok ? '' : 'disabled title="No matching brackets with the other active filters"'} onclick="setFilter('${kind}',${JSON.stringify(v)})">${name}</button>`; }).join('')}</div></div>`).join('')}
    <button class="clear-btn" onclick="clearFilters()">Clear all filters</button>`;
}

function setObjective(id) { state.objective = id; state.strategy = anyFilter() ? CUSTOM : id; state.notice = ''; writeHash(); render(); }
function setRisk(value) { state.risk = Math.max(0, Math.min(1, Number(value) / 100)); writeHash(); renderCompare(); renderBoard(); }
function setFilter(kind, value) {
  if (state.pick[kind] === value) state.pick[kind] = null; else state.pick[kind] = value;
  state.strategy = anyFilter() ? CUSTOM : state.objective; state.notice = ''; writeHash(); render();
}
function clearFilters() { state.pick = { champ: null, ones: null, depth: null, pred: null, src: null }; state.strategy = state.objective; state.notice = ''; writeHash(); render(); }

/* Leaderboard: the real track record embedded in this season's payload
 * (trackRecord() -- scripts/build_track_record.py), for the two backtested
 * rules and the fitted model. Not shown for a season that has not been
 * played -- there is no realisation to report yet -- and the rule search is
 * never a row here, on purpose: it is never scored against the pool (see
 * its own panel's copy), so a leaderboard position for it would be a number
 * this repo has explicitly refused to compute. */
function renderLeaderboard() {
  const box = document.getElementById('leaderboard');
  if (!box) return;
  const s = state.season;
  const tr = s && s.track_record;
  if (!tr || !tr.strategies) { box.hidden = false; box.innerHTML = '<p class="adjust-help">Historical results are available after a completed tournament has a recorded outcome.</p>'; return; }
  const LABEL = { p1: 'Win the pool', ev: 'Most expected points', model: 'Fitted model' };
  const rows = Object.entries(tr.strategies)
    .filter(([id]) => LABEL[id])
    .map(([id, r]) => ({ id, label: LABEL[id], ...r }))
    .sort((a, b) => b.won_share - a.won_share || b.points - a.points);
  if (!rows.length) { box.hidden = false; box.innerHTML = '<p class="adjust-help">No historical strategy results are available for this season.</p>'; return; }
  box.hidden = false;
  box.innerHTML = `
    <div class="panel-head"><p class="panel-title">Strategy leaderboard — ${s.year}, as played</p></div>
    <table class="lb">
      <thead><tr><th>Strategy</th><th class="num">Points</th><th class="num">Won share</th><th>Finish</th></tr></thead>
      <tbody>${rows.map(r => `
        <tr><td>${r.label}</td>
          <td class="num">${r.points.toLocaleString()}</td>
          <td class="num">${Math.round(r.won_share * 100)}%</td>
          <td>${finishText(r)}</td>
        </tr>`).join('')}
      </tbody>
    </table>
    <p class="lb-foot">Points scored against the real ${s.year} outcome; won share and finish are across the same
      ${tr.n_trials ? tr.n_trials.toLocaleString() : ''} simulated ${tr.pool_size || 30}-entry pools P(1st) is measured
      against. One season is one draw, so this is not a multi-year average. The rule search is not scored and has no row here.</p>`;
}

/* Header countdown, shown only while the newest listed season is
 * not_started (fieldPending()). No payload field carries an exact tip-off
 * date -- src/prediction/market_probabilities.py's own working assumption is
 * "Selection Sunday is typically mid-March; March 15 is conservative" -- so
 * this is stated as an estimate against that same assumption, never as a
 * precise clock. */
function renderCountdown() {
  const box = document.getElementById('countdown');
  if (!box) return;
  const s = state.season;
  if (!fieldPending(s) || !s) { box.hidden = true; return; }
  const est = new Date(Date.UTC(s.year, 2, 15)); // month 2 = March, 0-indexed
  const days = Math.ceil((est - new Date()) / 86400000);
  box.hidden = false;
  box.innerHTML = days > 0
    ? `~<b>${days}</b> days to Selection Sunday (est.)`
    : `Selection Sunday ${s.year} is imminent (est.)`;
}


function playedSeasonsBefore(year) {
  return (state.seasonsIndex || []).filter(x => x.status === 'ready' && x.year < year).map(x => x.year).sort((a, b) => a - b);
}

/* A listed season whose field is not out yet, with played seasons before
 * it: 2027 from the day 2026 is played until Selection Sunday. The rule
 * search has everything it needs -- it reads only prior seasons -- and only
 * the bracket waits for the field. The pool strategies and the fitted
 * model need the field itself, so under a pending season they are listed
 * and not selectable. */
function fieldPending(s = state.season) {
  return !!s && s.status === 'not_started' && playedSeasonsBefore(s.year).length > 0;
}

function setStrategy(id) {
  // Without a field there is nothing for any card to show; they are listed
  // and not selectable (strategyRows()'s fieldPending branch marks them
  // `pending`, and renderCompare() skips the click handler for those).
  if (fieldPending() && id !== RULE) return;
  if (![MODEL, RULE, 'p1', 'ev'].includes(id)) return;
  if (id === 'p1' || id === 'ev') state.objective = id;
  state.strategy = anyFilter() && id !== MODEL && id !== RULE ? CUSTOM : id;
  if (id === RULE) runRuleSearch();
  if (id === MODEL) { const panel = document.getElementById('explore-panel'); if (panel && panel.parentElement.tagName === 'DETAILS') panel.parentElement.open = true; }
  writeHash();
  refit();
  render();
}

function ruleKeys(season) {
  return [...Object.keys(season.z || {}), 'seed'].sort();
}
function ruleSeasonPayload(payload) {
  return { year: payload.year, first_round: payload.first_round, crit: { ...(payload.z || {}), seed: payload.teams.map(t => -t.seed) }, seed: payload.teams.map(t => t.seed), actual: payload.actual };
}
async function runRuleSearch() {
  const host = document.getElementById('rule-panel');
  if (!state.season || !state.seasonsIndex || typeof ruleSearchJob !== 'function') return;
  if (host && host.parentElement && host.parentElement.tagName === 'DETAILS') host.parentElement.open = true;
  state.rule.busy = true; state.rule.error = null; if (host) host.hidden = false; renderRulePanel();
  try {
    const years = state.seasonsIndex.filter(x => x.status === 'ready' && x.year < state.year).map(x => x.year).sort((a, b) => b - a);
    const raw = (await Promise.all(years.map(loadSeason))).filter(s => s && s.actual && s.first_round && s.z);
    const payloads = raw.map(ruleSeasonPayload);
    const here = raw.length ? ruleSeasonPayload(raw[0]) : null;
    const keys = ruleKeys(raw[0] || state.season);
    const result = ruleSearchJob({ played: payloads, here, keys, checkpoints: state.rule.checkpoints, maxCriteria: state.rule.maxCriteria, want: null });
    state.rule.result = result; state.rule.busy = false; renderRulePanel(); renderCompare();
  } catch (e) { state.rule.busy = false; state.rule.error = e.message || String(e); renderRulePanel(); }
}
function setRuleCheckpoint(round) {
  const set = new Set(state.rule.checkpoints); set.has(round) ? set.delete(round) : set.add(round);
  if (![1, 2, 3].some(r => set.has(r))) return;
  state.rule.checkpoints = [...set].sort((a, b) => a - b); writeHash(); runRuleSearch();
}
function setRuleComplexity(max) { state.rule.maxCriteria = max; writeHash(); runRuleSearch(); }
function setRuleChosen(i) { state.rule.chosen = i; writeHash(); renderRulePanel(); }
function renderRulePanel() {
  const box = document.getElementById('rule-panel'); if (!box) return;
  if (state.strategy !== RULE) { box.hidden = false; box.innerHTML = '<p class="adjust-help">Select the Rule search card above to run a historical reproducibility search.</p>'; return; }
  box.hidden = false;
  const r = state.rule.result;
  const rounds = RULE_CHECKPOINTS.map(x => `<button class="chip${state.rule.checkpoints.includes(x.r) ? ' on' : ''}" onclick="setRuleCheckpoint(${x.r})">${x.label}</button>`).join('');
  const controls = `<div class="rule-timeline"><b>Checkpoints</b>${RULE_CHECKPOINTS.map(x => `<button class="timeline-step${state.rule.checkpoints.includes(x.r) ? ' on' : ''}" onclick="setRuleCheckpoint(${x.r})"><span>${x.r + 1}</span>${x.label}</button>`).join('')}</div><div class="objective-row"><b>Complexity:</b><button class="chip${state.rule.maxCriteria === 2 ? ' on' : ''}" onclick="setRuleComplexity(2)">Simple (2)</button><span class="adjust-help">At most two variables across rounds.</span><button class="chip${state.rule.maxCriteria === 3 ? ' on' : ''}" onclick="setRuleComplexity(3)">Flexible (3)</button><span class="adjust-help">Up to three variables, more expressive but easier to overfit.</span></div>`;
  if (state.rule.busy) { box.innerHTML = `<h3>Rule search</h3>${controls}<p>Searching prior seasons…</p>`; return; }
  if (state.rule.error) { box.innerHTML = `<h3>Rule search</h3>${controls}<p>${state.rule.error}</p>`; return; }
  const entries = (r && r.entries) || [];
  const rows = entries.map((e, i) => `<button class="chip${i === state.rule.chosen ? ' on' : ''}" onclick="setRuleChosen(${i})"><b>${e.seq.map((k, n) => `${['R64','R32','S16','E8','F4','Final'][n]}: ${k}`).join(' · ')}</b><span>${e.matches.k}/${e.matches.m} older-season matches · ${e.complexity[0]} criteria · ${e.complexity[1]} switches</span></button>`).join('');
  const chosen = entries[state.rule.chosen];
  box.innerHTML = `<h3>Rule search</h3>${controls}<p class="adjust-help">A rule chooses the better team on one variable for each round. It is experimental and not pool-scored.</p>${entries.length ? `<p><b>Recent reproducibility run</b>: ${r.run.join(', ')} · ${r.nRules.toLocaleString()} surviving rules.</p><div class="rule-results">${rows}</div><p><b>Older-season matches</b>: the selected rule matches ${chosen.matches.k} of ${chosen.matches.m} seasons${chosen.matches.years.length ? ` (${chosen.matches.years.join(', ')})` : ''} outside the recent run.</p><p><b>Alternative rules</b>: choose another sequence above to compare its round-by-round record.</p>` : '<p>No rule matched those checkpoints in the available seasons.</p>'}`;
}

function setExplore(key) { state.explore = key; renderExplore(); }
function setExploreTab(tab) { state.exploreTab = tab; renderExplore(); }
function renderExplore() {
  const box = document.getElementById('explore-panel'); if (!box) return;
  if (state.strategy !== MODEL || !state.season || !fitReady()) { box.hidden = false; box.innerHTML = '<p class="adjust-help">Select the Fitted model card above to explore variables, rankings, matchup gaps, and sensitivity.</p>'; return; }
  box.hidden = false;
  const vars = state.season.variables || []; const key = vars.some(v => v.key === state.explore) ? state.explore : (vars[0] || {}).key;
  state.explore = key; const meta = vars.find(v => v.key === key) || { key, label: key };
  const values = (state.season.z && state.season.z[key]) || [];
  const ranked = state.season.teams.map((t, i) => ({ t, v: values[i] })).filter(x => Number.isFinite(x.v)).sort((a, b) => b.v - a.v).slice(0, 8);
  const groups = {}; vars.forEach(v => (groups[v.group || 'Other'] ||= []).push(v));
  const picker = Object.entries(groups).map(([group, items]) => `<div class="variable-group"><b>${group}</b><div class="chip-list">${items.map(v => `<button class="chip${v.key === key ? ' on' : ''}" onclick="setExplore('${v.key}')">${v.label || v.key}${CANONICAL_KEYS.includes(v.key) ? ' · in model' : ''}</button>`).join('')}</div></div>`).join('');
  const top = ranked.map(x => `<div class="scard-row"><span>${x.t.seed} ${x.t.name}</span><b>${x.v.toFixed(2)}σ</b></div>`).join('');
  const games = solveByFit().flat().map(g => ({ ...g, gap: Math.abs((values[g.a] || 0) - (values[g.b] || 0)) })).sort((a, b) => b.gap - a.gap).slice(0, 5);
  const matchup = games.map(g => `<div class="scard-row"><span>${state.season.teams[g.a].name} vs ${state.season.teams[g.b].name}</span><b>${g.gap.toFixed(2)}σ · model: ${state.season.teams[g.win].name} (${Math.round(g.p * 100)}%)</b></div>`).join('');
  let signal = '';
  const col = state.training && state.training.keys.indexOf(key);
  if (col >= 0 && typeof variableRecord === 'function') {
    const rec = variableRecord(state.training.games, col, state.year);
    if (rec) signal = `<p class="adjust-help">Historical one-variable signal: the higher-rated side won ${Math.round(rec.betterWins.rate * 100)}% of ${rec.betterWins.n} games; gap correlation with margin ${rec.corr.toFixed(2)}.</p>`;
  }
  const tabs = `<div class="objective-row"><button class="chip${state.exploreTab === 'field' ? ' on' : ''}" onclick="setExploreTab('field')">Field ranking</button><button class="chip${state.exploreTab === 'matchups' ? ' on' : ''}" onclick="setExploreTab('matchups')">Matchup gaps</button><button class="chip${state.exploreTab === 'sensitivity' ? ' on' : ''}" onclick="setExploreTab('sensitivity')">Model sensitivity</button></div>`;
  const higher = meta.higher_better === false ? 'Lower values are better (for example, rank). Higher values are better for this variable.' : 'Higher values are better for this variable.';
  const sensitivity = state.fit.keys.includes(key) ? `<p class="adjust-help">This variable is used by the fitted model. Its coefficient is ${state.fit.beta[state.fit.keys.indexOf(key)].toFixed(2)} margin points per standard deviation; removing variables requires a full refit.</p>` : '<p class="adjust-help">This variable is available for exploration but is not used by the validated fitted model.</p>';
  const body = state.exploreTab === 'field' ? `<p><b>${meta.label || key}</b> — ${higher}</p>${top}${signal}` : state.exploreTab === 'matchups' ? `<p><b>Largest matchup gaps on the fitted bracket</b></p>${matchup}` : sensitivity;
  box.innerHTML = `<h3>Explore fitted model variables</h3><p class="adjust-help">Inspect rankings, historical signal, matchup gaps, and sensitivity. Choosing a variable here does not change the fitted bracket.</p>${tabs}<div class="variable-picker">${picker}</div>${body}`;
}

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/* ---------- controls ---------- */

/* Loads the one season the page shows. There is no year nav (removed
 * 2026-09-19 along with the rule search, the adjust filters, the team
 * drawer and the methodology panel -- see index2.html): pickDefaultSeason()
 * in init() decides which season that is, and it never changes after
 * load. */
async function setYear(year) {
  state.year = year;
  try {
    state.season = await loadSeason(year);
  } catch {
    state.season = null;
  }
  // Refit: the excluded season changed, so the coefficients must change too.
  refit();
  writeHash();
  render();
}

async function init() {
  const [idx] = await Promise.all([
    fetch(`data/seasons.json?v=${DATA_V}`).then(r => r.json()),
    loadTraining(),
  ]);
  // pickDefaultSeason(): the newest listed season's bracket once it has one,
  // else the newest thing on record while its field is pending.
  state.seasonsIndex = idx.seasons;
  readHash();
  const fallback = pickDefaultSeason(idx.seasons);
  if (fallback !== null) state.year = fallback;

  await setYear(state.year);
}

init();
