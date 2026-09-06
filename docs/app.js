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
 * THE FIRST TWO ARE NOT THE SAME BRACKET AND THE DIFFERENCE IS THE POINT. In
 * 2026 they name different champions: the win-maximising bracket gives up 69
 * expected points (872 against 941) to roughly two-and-a-half times its win
 * probability (9.9% against 3.9%). In a winner-take-all pool the points number
 * is worth nothing and the trade is free; in a pool paying second and third it
 * is a real decision. Both scores are shown for whichever strategy is selected,
 * so the cost is visible rather than implied.
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
const DATA_V = 15;

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
  if (!src || !wanted.length) { state.fit = null; return; }

  // Variables the matrix cannot supply are dropped, not zero-filled: a zero
  // differential is a claim that the two teams are equal on it.
  const cols = [];
  const keys = [];
  for (const k of wanted) {
    const i = src.keys.indexOf(k);
    if (i >= 0) { keys.push(k); cols.push(i); }
  }
  if (!keys.length) { state.fit = null; return; }

  const f = fitLinear(src.games, cols, state.year);
  f.keys = keys;
  f.cols = cols;
  f.userKeys = keys;
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
  const teams = state.season.teams;
  let current = state.season.first_round.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const p = winProb(a, b);
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
  const se = obj === 'p1' ? ((state.season.filters || {}).p1_se || 0) : 0;
  const cut = sorted[0][obj] - se;
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
              ? ` Showing ${Math.min(state.alt, alts.length - 1) + 1} of ${alts.length} within the referee's margin.`
              : ''),
      picks: decodeBracket(src.b), ev: src.ev, p1: src.p1,
      alts: alts.length, altIndex: Math.min(state.alt, alts.length - 1),
      source: SRC_LABEL[src.s] || src.s,
    };
  }
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
    state.rounds = null;
    { const tools = document.getElementById('board-tools'); if (tools) tools.hidden = true; }
    weights.hidden = true;
    { for (const id of ['champions', 'ones', 'shapes', 'dd16', 'sources', 'alts']) {
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
    const kind = !st ? 'LOYO validated'
      : st.id === CUSTOM ? 'Your pick'
      : st.id === 'ev' ? 'Exact optimum'
      : 'Backtested rule';
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

  // The disclosure rides with the numbers. Shown whenever a P(1st) figure is on
  // screen, which is every "% to win" on the strategy cards and every chip.
  const p1note = document.getElementById('p1-note');
  if (p1note) p1note.textContent = s.p1_assumption || '';

  document.getElementById('equation').innerHTML = anyEnabled() ? equationHTML() : '';

  const rounds = usingOptimized() ? solveFromPicks() : solveByFit();
  const truth = solveActual();
  // Kept for the exporter. Presentation state only -- copyPicks() serialises
  // exactly what is on screen rather than re-deriving it, so the two can never
  // disagree about which bracket the user is looking at.
  state.rounds = rounds;
  { const tools = document.getElementById('board-tools'); if (tools) tools.hidden = false; }
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
      return `${(v.p1 * 100).toFixed(1)}% to win · ${v.ev.toFixed(0)} pts`
           + (filt ? ' · filtered' : '');
    })(),
  }));
  opts.push({
    id: MODEL,
    label: 'Fitted model',
    sub: 'Ridge regression fitted here on tournament games from seasons before this one. '
       + 'Chosen by measurement, not configurable: it beat kNN, LightGBM and local linear on the '
       + 'same walk-forward split. The only strategy the history slider applies to.',
    tag: 'live',
    stat: state.fit && state.fit.oos
      ? `${(state.fit.oos.accuracy * 100).toFixed(1)}% out-of-sample`
      : '',
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
  const lead = [
    'Each is the best bracket available with that team winning.',
    'How many 1 seeds reach the Final Four. Depth says how far DOWN you reach; this says how much of the top you keep.',
    'The deepest seed your Final Four reaches.',
    'Bracket shapes the pipeline already scores, from src/product/selection.py.',
    'Which model imagined the bracket. These disagree about real teams, which is the point.',
  ];
  notes.forEach((el, i) => { if (el) el.textContent = `${lead[i]} ${suffix}`; });
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

    host.innerHTML = values.map(v => {
      const on = state.pick[row.kind] === v;
      // Availability is a query, not a table: does anything survive with this
      // value plus whatever else is selected.
      const rows = inert ? [] : matching({ ...state.pick, [row.kind]: v });
      const ok = rows.length > 0;
      const best = ok ? rows.reduce((a, b) => (b[obj] > a[obj] ? b : a)) : null;
      const stat = !best ? '—'
        : obj === 'ev' ? `${best.ev.toFixed(0)} pts` : `${(best.p1 * 100).toFixed(1)}%`;

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
    const stat = obj === 'ev' ? `${r.ev.toFixed(0)} pts` : `${(r.p1 * 100).toFixed(1)}%`;
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
function picksAsText() {
  if (!state.rounds || !state.season) return '';
  const st = usingOptimized() ? currentStrategy() : null;
  // g.win is an INDEX into season.teams, the same thing sideHTML() resolves.
  // Treating it as a team id silently produced "undefined 8" for every pick.
  const team = i => state.season.teams[i] || {};

  const head = [
    `${state.year} bracket — ${st ? st.label : 'Fitted model'}`,
    st && st.p1 !== undefined
      ? `${(st.p1 * 100).toFixed(1)}% to finish first, ${st.ev.toFixed(0)} expected points`
      : '',
    // The disclosure travels with the picks. A bracket pasted into a group chat
    // outlives the page it came from, and the number goes with it.
    state.season.p1_assumption || '',
  ].filter(Boolean);

  const body = state.rounds.map((games, r) => {
    const winners = games.map(g => { const t = team(g.win); return `${t.seed} ${t.name}`; });
    return `${ROUNDS[r]}\n${winners.map(w => `  ${w}`).join('\n')}`;
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
  if (!matching().length) {
    for (const other of ['pred', 'src', 'depth', 'ones', 'champ']) {
      if (other === kind || state.pick[other] === null) continue;
      state.pick[other] = null;
      if (matching().length) break;
    }
  }
  if (!matching().length) {
    state.pick = { champ: null, ones: null, depth: null, pred: null, src: null };
    state.pick[kind] = value;
  }
  if (!matching().length) state.pick = prev;

  state.strategy = anyFilter() ? CUSTOM : state.objective;
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
    if (id === MODEL) state.pick = { champ: null, ones: null, depth: null, pred: null, src: null };
    state.strategy = id;
  }
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
    fetch(`data/seasons.json?v=${DATA_V}`).then(r => r.json()),
    loadTraining(),
  ]);
  // Open on the newest season that actually has a bracket. Not max(year):
  // 2027 is listed from the moment the calendar knows about it and stays
  // "not_started" until Selection Sunday, so the newest LISTED season is an
  // empty state for most of the year.
  const ready = idx.seasons.filter(s => s.status === 'ready').map(s => s.year);
  if (ready.length) state.year = Math.max(...ready);

  document.getElementById('years').innerHTML = idx.seasons.map(s => `
    <button class="yr${s.year === state.year ? ' on' : ''}${s.status === 'ready' ? '' : ' na'}"
            data-year="${s.year}" title="${s.status === 'ready' ? '' : `No bracket available for ${s.year}`}"
            onclick="setYear(${s.year})">${s.year}</button>`).join('');

  document.getElementById('d-close').addEventListener('click', closeDrawer);
  document.getElementById('scrim').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  await setYear(state.year);

  // The row scrolls on a narrow viewport and the default season sits at the
  // far right of seventeen, so it would otherwise open out of view.
  document.querySelector('.yr.on')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

// Track which team the drawer is showing so a weight change can refresh it.
const _openTeam = openTeam;
openTeam = function (i) { state.openIdx = i; _openTeam(i); };

init();
