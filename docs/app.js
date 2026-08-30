/* Bracket Lab — one page, one question: what are you optimising for.
 *
 *   Maximise chance of winning   the bracket with the highest P(finishing
 *                                first), chosen in Python by the LOYO-validated
 *                                selector and shipped precomputed.
 *
 *   Maximise expected points     the bracket with the highest expected ESPN
 *                                score, from the same validated selector.
 *
 *   Fitted model                 a ridge or nearest-neighbour SPREAD regression
 *                                fitted live on real tournament games. Predicts
 *                                scoring MARGIN in points, with P(win) following
 *                                as Phi(margin / sigma); see fit.js. It is not a
 *                                classifier and a coefficient is not a log-odds.
 *                                The only strategy the history slider applies to.
 *
 * THE FIRST TWO ARE NOT THE SAME BRACKET AND THE DIFFERENCE IS THE POINT. In
 * 2026 they name different champions: the P(1st) bracket gives up 195 expected
 * points (727 against 922) to roughly triple its win probability (10.2% against
 * 3.1%). In a winner-take-all pool the second number is worth nothing and the
 * trade is free; in a pool paying second and third it is a real decision. Both
 * scores are shown for whichever strategy is selected so the cost is visible
 * rather than implied.
 *
 * WHAT USED TO BE HERE. The page let the user switch individual variables on
 * and off and fitted whatever they chose. That was removed: it invited people to
 * build models nobody had validated, and measurement said the choosing bought
 * nothing (per-fold feature selection scored 0.46651 against 0.45698 for the
 * fixed canonical set, inside the bootstrap's noise). Choosing an objective is a
 * decision the data cannot make for you; choosing predictors is one it can.
 *
 * The fit excludes the displayed season (leave-one-year-out), so the
 * coefficients were never derived from the games being predicted.
 */

const ROUNDS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];

/* The browser-fitted strategy. Anything else is a precomputed bracket read out
 * of the season payload by id. */
const MODEL = 'model';

/* Variables the fitted strategy uses.
 *
 * FIXED, NOT CHOSEN. This is the key set the frozen baseline in
 * artifacts/model_baseline.json is defined over, and it is what the shipped
 * accuracy number (log loss 0.45296 on held-out tournament games) describes.
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
  year: 2026,
  strategy: 'p1',       // a payload strategy id, or MODEL
  fit: null,            // {beta, n, converged}
  training: null,
  season: null,
  priors: null,        // historical seed-matchup upset rates, per season
  priorWeight: 0,      // 0 = model only; the blend control's rest position
  model: 'ridge',      // 'ridge' | 'knn'
  k: 25,               // neighbours, when model === 'knn' 
  cache: {},
};

/* ---------- data ---------- */

/* Historical seed-matchup upset rates, built walk-forward per season by
 * scripts/build_upset_priors.py. Null until loaded, and the blend degrades to
 * the model alone if it never arrives.
 *
 * BUMP ?v= WHENEVER THE FILE'S CONTENTS CHANGE. It is the only cache-busting
 * mechanism here -- there is no service worker and the filename is stable -- so
 * a returning browser will keep serving whatever it cached against the old URL.
 * v=2 is the 2010-2025 window; v=1 was built from 1985 onward, and the two
 * disagree by enough to matter (6-11 upsets .380 vs .489). A stale cache would
 * not error, it would quietly predict with the previous decade's priors. */
async function loadPriors() {
  if (state.priors) return state.priors;
  try {
    const res = await fetch('data/upset_priors.json?v=2');
    state.priors = await res.json();
  } catch {
    state.priors = {};
  }
  return state.priors;
}

/* The regular-season + conference-tournament matrix: 41,321 rows against the
 * tournament set's 1,008. Loaded lazily -- it is 9 MB and most users never
 * switch to it. */
async function loadPit() {
  if (state.pit !== undefined) return state.pit;
  try {
    const res = await fetch('data/training_pit.json?v=1');
    state.pit = await res.json();
  } catch {
    state.pit = null;
  }
  return state.pit;
}

/* Per-variable scale factor carrying a season payload's differential onto the
 * regular-season matrix's scale.
 *
 * WHY THIS IS NEEDED AT ALL. The payload standardises within the 68-team
 * tournament field; training_pit standardises within the ~350-team D1 field at
 * each week boundary. A differential cancels the location shift but not the
 * scale, so dz_D1 = dz_68 * (sd_68 / sd_D1) -- ratios run 1.05 to 1.29 here.
 * Feeding an unconverted query to coefficients fitted on the other scale would
 * silently shrink every prediction, and the board would simply look
 * under-confident rather than wrong. */
function pitScale(key) {
  if (!state._scale) state._scale = {};
  if (state._scale[key] !== undefined) return state._scale[key];
  const sd = (rows, keys) => {
    const i = keys.indexOf(key);
    if (i < 0) return null;
    let m = 0; for (const r of rows) m += r.x[i]; m /= rows.length;
    let v = 0; for (const r of rows) v += (r.x[i] - m) ** 2;
    return Math.sqrt(v / rows.length);
  };
  const a = sd(state.training.games, state.training.keys);
  const b = state.pit ? sd(state.pit.games, state.pit.keys) : null;
  state._scale[key] = (a && b && b > 1e-9) ? a / b : 1;
  return state._scale[key];
}

async function loadTraining() {
  if (state.training) return state.training;
  const res = await fetch('data/training.json?v=1');
  state.training = await res.json();
  return state.training;
}

async function loadSeason(year) {
  if (state.cache[year]) return state.cache[year];
  const res = await fetch(`data/season_${year}.json?v=1`);
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
const VENUE_KEYS = ['venue_home', 'venue_host_city', 'venue_travel'];

function refit() {
  const wanted = CANONICAL_KEYS;
  if (!state.training || !wanted.length) { state.fit = null; return; }

  const regular = state.trainingSet === 'regular' && state.pit;
  const src = regular ? state.pit : state.training;

  // Variables the chosen matrix cannot supply are dropped, not zero-filled: a
  // zero differential is a claim that the two teams are equal on it.
  const keys = wanted.filter(k => src.keys.indexOf(k) >= 0);
  if (!keys.length) { state.fit = null; return; }

  // VENUE RIDES ALONG WHEN FITTING ON REGULAR-SEASON ROWS, always, and is not
  // user-selectable. Those rows are mostly home-or-away games, and omitting
  // venue pushes the home effect into the correlated strength coefficients --
  // measured at +13% on srs_blend and +44% on barthag, because strong teams
  // buy home games. The coefficients would then be applied to all-neutral
  // tournament games and over-predict. The terms are zeroed at prediction
  // instead; see diffVector.
  const extra = regular ? VENUE_KEYS.filter(k => src.keys.indexOf(k) >= 0) : [];
  const fitKeys = keys.concat(extra);
  const cols = fitKeys.map(k => src.keys.indexOf(k));

  const f = fitLinear(src.games, cols, state.year);
  f.keys = fitKeys;
  f.cols = cols;
  f.userKeys = keys;
  f.regular = regular;
  f.dropped = wanted.filter(k => src.keys.indexOf(k) < 0);   // e.g. t_rank has no dated snapshot
  f.quality = fitQuality(state.training.games, cols, state.year, f.beta);
  // The honest number: fit on prior seasons, scored on seasons never seen.
  f.oos = crossValidate(state.training.games, cols, state.training.years, 2014);
  state.fit = f;
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
    if (VENUE_KEYS.includes(k)) return 0;
    const col = z[k];
    const d = col ? (col[a] || 0) - (col[b] || 0) : 0;
    return f.regular ? d * pitScale(k) : d;
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
  if (state.model === 'knn') {
    // The kNN board answers a different question -- what happened in the games
    // that looked most like this one -- so it gets its own margin and its own
    // LOCAL sigma from the spread of the neighbours that voted.
    const src = state.fit.regular ? state.pit : state.training;
    const r = knnPredict(src.games, state.fit.cols, diffVector(a, b), state.k, state.year);
    if (r) return winProbFromMargin(r.margin, r.sigma, cal);
  }
  return winProbFromMargin(margin(a, b), state.fit.sigma, cal);
}

const ROUND_KEYS = ['R64', 'R32', 'S16', 'E8', 'F4', 'NCG'];

/* P(team a beats team b) from the historical base rate for their seed pairing,
 * or NaN when there is no cell for it.
 *
 * The stored rate is always P(the WORSE seed wins), so it has to be flipped
 * when `a` is the better seed. Same-seed matchups have no upset to speak of
 * and return NaN, which blendWithPrior passes through untouched. */
function priorFor(a, b, roundIdx) {
  const tbl = state.priors && state.priors[String(state.year)];
  if (!tbl) return NaN;
  const teams = state.season.teams;
  const sa = teams[a].seed, sb = teams[b].seed;
  if (!sa || !sb || sa === sb) return NaN;
  const better = Math.min(sa, sb), worse = Math.max(sa, sb);
  const cell = (tbl.cells[ROUND_KEYS[roundIdx]] || {})[`${better}-${worse}`];
  if (!cell) return NaN;
  return sa === worse ? cell.p : 1 - cell.p;
}

/* Play the bracket out under the fit. Exact ties go to the better seed, then
 * lower index, so the board never jitters on a coin-flip game. */
function solveByFit() {
  const teams = state.season.teams;
  let current = state.season.first_round.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const p = blendWithPrior(winProb(a, b), priorFor(a, b, r), state.priorWeight);
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
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const win = picks[r].has(a) ? a : b;
      games.push({ a, b, win, sa: null, sb: null });
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
function currentStrategy() {
  const list = (state.season && state.season.strategies) || [];
  return list.find(s => s.id === state.strategy) || null;
}

function usingOptimized() {
  return state.strategy !== MODEL;
}

function anyEnabled() {
  return !usingOptimized() && state.fit && state.fit.keys.length > 0;
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
    weights.hidden = true;
    { for (const id of ['prior-panel', 'model-panel']) {
        const el = document.getElementById(id); if (el) el.hidden = true; } }
    note.innerHTML = '';
    empty.hidden = false;
    empty.innerHTML = `
      <p class="e-title">${s ? s.message : 'Season unavailable.'}</p>
      <p class="e-sub">${s ? s.detail : ''}</p>`;
    return;
  }

  empty.hidden = true;
  weights.hidden = false;   // the panel is the only control surface
  // The prior blend applies to the fitted board only. The Optimized picks are
  // precomputed and are not a regression, so there is nothing to blend into.
  { const fitted = !usingOptimized();
    const pp = document.getElementById('prior-panel');
    const mp = document.getElementById('model-panel');
    if (pp) pp.hidden = !fitted;
    if (mp) mp.hidden = !fitted; }

  if (usingOptimized()) {
    const st = currentStrategy();
    // Both scores, always, for whichever strategy is showing. A bracket built to
    // win outright gives up real expected points to do it, and stating only the
    // number its own objective optimises would hide exactly that cost.
    const kind = st && st.id === 'ev' ? 'Exact optimum' : 'LOYO validated';
    note.innerHTML = `<span class="tag">${kind}</span><span>${st ? st.note : s.pool_optimized_note}` +
      (st ? ` <strong>${(st.p1 * 100).toFixed(1)}%</strong> chance of finishing first, ` +
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
           `(RMSE ${o.rmse.toFixed(1)}) and calls <strong>${(o.accuracy * 100).toFixed(1)}%</strong> of them correctly ` +
           `— against ${(f.quality.accuracy * 100).toFixed(1)}% on the games it was fitted to.` +
           // Accuracy grades the pick; the board also shows a percentage, and
           // that is a separate claim needing a separate number.
           (o.probScore ? ` The percentages themselves score <strong>Brier ${o.probScore.brier.toFixed(3)}</strong> ` +
             `(log loss ${o.probScore.logLoss.toFixed(3)}), after calibrating the margin-to-probability link on those ` +
             `held-out games${o.calibration ? ` — Student-t, ν=${o.calibration.nu === Infinity ? '∞' : o.calibration.nu}, scale ${o.calibration.a.toFixed(2)}` : ''}.` : '')
         : `Not enough history to test out-of-sample.`) +
      `</span>`;
  }

  document.getElementById('equation').innerHTML = anyEnabled() ? equationHTML() : '';

  const rounds = usingOptimized() ? solveFromPicks() : solveByFit();
  const truth = solveActual();
  board.innerHTML = rounds.map((games, r) => `
    <div class="round" style="--n:${games.length}">
      <p class="r-label">${ROUNDS[r]}</p>
      ${games.map((g, gi) => gameHTML(g, r, truth ? truth[r][gi] : null)).join('')}
    </div>`).join('');
}

/* The fitted model, written out.
 *
 * Terms are ordered by magnitude rather than by menu position, so the variables
 * actually carrying the model come first. Every delta is a difference in
 * standard deviations between the two teams, which is why a coefficient reads as
 * POINTS OF MARGIN per standard deviation of edge.
 */
function equationHTML() {
  const f = state.fit;
  const label = Object.fromEntries(state.season.variables.map(v => [v.key, v.label]));

  // A coefficient that changes sign between walk-forward folds is not a
  // finding, however large it looks. Marking those is the difference between
  // showing the model and vouching for it.
  const stab = f.oos ? f.oos.stability : null;

  const terms = f.keys
    .map((k, i) => ({ k, b: f.beta[i], label: label[k] || k, s: stab ? stab[i] : null }))
    .sort((a, b) => Math.abs(b.b) - Math.abs(a.b));

  const body = terms.map((t, i) => {
    const sign = t.b < 0 ? '\u2212' : '+';
    const mag = Math.abs(t.b).toFixed(2);
    const weak = Math.abs(t.b) < 0.05;
    const shaky = t.s && t.s.signFlips;
    const cls = weak ? ' weak' : (shaky ? ' shaky' : '');
    const tip = weak ? 'Essentially no contribution'
      : shaky ? `Unstable: ranged ${t.s.min.toFixed(1)} to ${t.s.max.toFixed(1)} across held-out seasons, changing sign. Do not read this number as an effect.`
      : '';
    return `<span class="term${cls}" title="${tip}">` +
           `${i === 0 && t.b >= 0 ? '' : `<i class="op">${sign}</i>`}` +
           `<b>${mag}</b><span class="dv">\u0394${t.label}</span>` +
           `${shaky ? '<i class="warn" aria-label="unstable">*</i>' : ''}</span>`;
  }).join('');

  const nShaky = terms.filter(t => t.s && t.s.signFlips).length;

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
        No intercept: swapping the teams flips the sign exactly.
        A positive margin is the pick; typical error is
        \u00b1${f.oos ? f.oos.mae.toFixed(1) : f.sigma.toFixed(1)} points.
      </p>
      ${nShaky ? `<p class="eq-warn">
        <i class="warn">*</i> ${nShaky} of these ${terms.length} coefficients change sign
        between held-out seasons. The equation as a whole still predicts \u2014
        that is what the out-of-sample figure measures \u2014 but those individual
        numbers are splitting credit between variables that overlap, and are
        not readable as "what this variable is worth". The variable set is fixed
        because dropping the redundant ones was measured and did not predict any
        better — it only made the coefficients easier to read.
      </p>` : ''}
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
      ${p === null ? '' : `<span class="sc">${Math.round(p * 100)}%</span>`}
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

  // "searched" and "exact" are different claims and the tag should not blur
  // them. P(1st) has no closed form -- it depends on the whole opponent field --
  // so it is the best of ~3,000 scored candidates. Expected points does have
  // one, solved exactly on the bracket, and calling that a search would
  // understate it while calling the other exact would overstate it.
  const opts = (s.strategies || []).map(st => ({
    id: st.id, label: st.label, sub: st.note,
    tag: st.id === 'ev' ? 'exact' : 'searched',
    stat: `${(st.p1 * 100).toFixed(1)}% to win · ${st.ev.toFixed(0)} pts`,
  }));
  opts.push({
    id: MODEL,
    label: 'Fitted model',
    sub: 'Fits a ridge or nearest-neighbour model here, on games from seasons before this one. '
       + 'The only strategy the history slider applies to.',
    tag: 'live',
    stat: state.fit && state.fit.oos
      ? `${(state.fit.oos.accuracy * 100).toFixed(1)}% out-of-sample`
      : '',
  });

  document.getElementById('strat-list').innerHTML = opts.map(o => `
    <label class="vopt${state.strategy === o.id ? ' active' : ''}">
      <input type="radio" name="strategy" ${state.strategy === o.id ? 'checked' : ''}
             onchange="setStrategy('${o.id}')">
      <span class="vopt-name">${o.label}</span>
      <span class="v-tag">${o.tag}</span>
      <span class="vopt-sub">${o.sub}</span>
      ${o.stat ? `<span class="vopt-stat">${o.stat}</span>` : ''}
    </label>`).join('');
}

function setStrategy(id) {
  state.strategy = id;
  refit();
  renderStrategies();
  render();
}

/* ---------- team drawer ---------- */

function openTeam(i) {
  const s = state.season, t = s.teams[i];
  document.getElementById('d-name').textContent = t.name;
  document.getElementById('d-sub').textContent = `${t.seed} seed · ${t.region}`;

  const groups = {};
  for (const v of s.variables) (groups[v.group] ||= []).push(v);

  document.getElementById('d-body').innerHTML = Object.entries(groups).map(([g, vars]) => `
    <div class="d-group">
      <p class="g-name">${g}</p>
      ${vars.map(v => {
        const z = (s.z[v.key] || [])[i] || 0;
        const raw = (s.raw[v.key] || [])[i];
        const pct = Math.max(2, Math.min(98, 50 + z * 16));
        // Lit means "this one is in the fitted model", which is now a fact to
        // read rather than a control to operate. Every stat is still shown,
        // because the drawer is for understanding a team, not for configuring
        // a model.
        const on = CANONICAL_KEYS.indexOf(v.key) >= 0;
        return `
        <div class="d-row${on ? ' lit' : ''}">
          <span class="d-lab">${v.label}</span>
          <span class="d-track"><i style="left:${pct}%"></i></span>
          <span class="d-val">${raw === null || raw === undefined ? '—' : fmt(raw)}</span>
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

/* Say plainly what the regular-season set costs. It is 41x the rows, and it is
 * also measurably worse on held-out tournament games (+0.124 log loss in the
 * step 3 measurement): it cannot carry t_rank, which has no dated snapshot and
 * is the model's most valuable single variable, and its wider standardisation
 * compresses tournament differentials. Offering it without saying so would
 * present a downgrade as a free choice. */
function updateTsetNote() {
  const el = document.getElementById('tset-note');
  if (!el) return;
  const f = state.fit;
  if (state.trainingSet !== 'regular') {
    el.textContent = '1,008 NCAA tournament games. Every variable is available.';
    return;
  }
  if (!state.pit) { el.textContent = 'Regular-season matrix unavailable.'; return; }
  const dropped = f && f.dropped && f.dropped.length ? f.dropped.join(', ') : 'none';
  el.innerHTML =
    `${state.pit.n_games.toLocaleString()} regular-season and conference-tournament games. ` +
    `Unavailable here: <b>${dropped}</b>. Venue is fitted and then zeroed for the ` +
    `neutral court. Measured worse than tournament-only on held-out games.`;
}

/* ---------- controls ---------- */

async function setYear(year) {
  state.year = year;
  document.querySelectorAll('.yr').forEach(b => b.classList.toggle('on', Number(b.dataset.year) === year));
  try {
    state.season = await loadSeason(year);
  } catch {
    state.season = null;
  }
  // Refit: the excluded season changed, so the coefficients must change too.
  refit();
  renderStrategies();
  render();
}

async function init() {
  const [idx] = await Promise.all([
    fetch('data/seasons.json?v=1').then(r => r.json()),
    loadTraining(),
    loadPriors(),
  ]);
  document.getElementById('years').innerHTML = idx.seasons.map(s => `
    <button class="yr${s.year === state.year ? ' on' : ''}" data-year="${s.year}"
            onclick="setYear(${s.year})">${s.year}</button>`).join('');

  document.querySelectorAll('input[name=tset]').forEach(el => {
    el.addEventListener('change', async e => {
      state.trainingSet = e.target.value;
      if (state.trainingSet === 'regular') await loadPit();
      refit(); render();
      updateTsetNote();
    });
  });
  document.querySelectorAll('input[name=model]').forEach(el => {
    el.addEventListener('change', e => {
      state.model = e.target.value;
      document.getElementById('k-row').hidden = state.model !== 'knn';
      document.getElementById('model-note').textContent = state.model === 'knn'
        ? `Each pick is the mean margin of the ${state.k} most similar prior tournament games, with its own spread as the uncertainty.`
        : 'Ridge fits one set of coefficients across every prior tournament game.';
      render();
    });
  });
  const kEl = document.getElementById('knn-k');
  if (kEl) {
    kEl.addEventListener('input', e => {
      state.k = Number(e.target.value);
      document.getElementById('knn-v').textContent = e.target.value;
      if (state.model === 'knn') {
        document.getElementById('model-note').textContent =
          `Each pick is the mean margin of the ${state.k} most similar prior tournament games, with its own spread as the uncertainty.`;
        render();
      }
    });
  }
  const priorEl = document.getElementById('prior-w');
  if (priorEl) {
    priorEl.addEventListener('input', e => {
      state.priorWeight = Number(e.target.value) / 100;
      document.getElementById('prior-v').textContent = `${e.target.value}%`;
      render();   // the blend changes picks, so the whole board is restated
    });
  }
  updateTsetNote();
  document.getElementById('d-close').addEventListener('click', closeDrawer);
  document.getElementById('scrim').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  await setYear(state.year);
}

// Track which team the drawer is showing so a weight change can refresh it.
const _openTeam = openTeam;
openTeam = function (i) { state.openIdx = i; _openTeam(i); };

init();
