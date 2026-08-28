/* Bracket Lab — one page, two ways to fill a bracket.
 *
 *   Optimized   the bracket chosen in Python by the LOYO-validated method.
 *               Shipped precomputed in the season payload; the browser renders
 *               it and does not re-derive it.
 *
 *   Fit         the user switches variables on and off; a ridge SPREAD
 *               regression is fitted live on real tournament games and its
 *               coefficients decide every matchup. Nobody has to guess a weight
 *               or a sign -- the data supplies both, and they are shown.
 *               It predicts scoring MARGIN in points, and P(win) follows as
 *               Phi(margin / sigma); see fit.js. It is not a classifier, and a
 *               coefficient is not a log-odds.
 *
 * The fit excludes the displayed season (leave-one-year-out), so the
 * coefficients were never derived from the games being predicted.
 */

const ROUNDS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];

const OPTIMIZED = '__optimized__';

const state = {
  year: 2026,
  enabled: new Set([OPTIMIZED]),   // variable keys, or OPTIMIZED
  fit: null,            // {beta, n, converged}
  training: null,
  season: null,
  priors: null,        // historical seed-matchup upset rates, per season
  priorWeight: 0,      // 0 = model only; the blend control's rest position
  cache: {},
};

/* ---------- data ---------- */

/* Historical seed-matchup upset rates, built walk-forward per season by
 * scripts/build_upset_priors.py. Null until loaded, and the blend degrades to
 * the model alone if it never arrives. */
async function loadPriors() {
  if (state.priors) return state.priors;
  try {
    const res = await fetch('data/upset_priors.json?v=1');
    state.priors = await res.json();
  } catch {
    state.priors = {};
  }
  return state.priors;
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
function refit() {
  const keys = [...state.enabled].filter(k => k !== OPTIMIZED);
  if (!state.training || !keys.length) { state.fit = null; return; }

  const cols = keys.map(k => state.training.keys.indexOf(k)).filter(i => i >= 0);
  if (!cols.length) { state.fit = null; return; }

  const f = fitLinear(state.training.games, cols, state.year);
  f.keys = keys;
  f.quality = fitQuality(state.training.games, cols, state.year, f.beta);
  // The honest number: fit on prior seasons, scored on seasons never seen.
  f.oos = crossValidate(state.training.games, cols, state.training.years, 2014);
  state.fit = f;
}

/* Predicted scoring margin for team a against team b, in points.
 *
 * Antisymmetric by construction: swapping a and b negates the differential and
 * so negates the margin exactly. */
function margin(a, b) {
  const z = state.season.z, f = state.fit;
  let t = 0;
  for (let j = 0; j < f.keys.length; j++) {
    const col = z[f.keys[j]];
    if (col) t += f.beta[j] * ((col[a] || 0) - (col[b] || 0));
  }
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
  return winProbFromMargin(margin(a, b), state.fit.sigma, state.fit.oos && state.fit.oos.calibration);
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

/* Expand the precomputed Optimized picks into the same shape. */
function solveFromPicks() {
  const picks = state.season.pool_optimized.map(r => new Set(r));
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

/* The prebuilt bracket and the fitted one are alternatives, not layers: one is
 * chosen by a validated pipeline, the other is fitted from whatever the user
 * enabled. Selecting either clears the other. */
function usingOptimized() {
  return state.enabled.has(OPTIMIZED);
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
  const weights = document.getElementById('weights');
  if (!s || s.status !== 'ready') {
    board.innerHTML = '';
    weights.hidden = true;
    { const pp = document.getElementById('prior-panel'); if (pp) pp.hidden = true; }
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
  { const pp = document.getElementById('prior-panel');
    if (pp) pp.hidden = state.enabled.has(OPTIMIZED); }

  if (usingOptimized()) {
    note.innerHTML = `<span class="tag">LOYO validated</span><span>${s.pool_optimized_note}</span>`;
  } else if (!anyEnabled()) {
    note.innerHTML = `<span class="tag alt">Pick variables</span><span>Switch on any variables above. A model is fitted to real tournament games and its coefficients decide every matchup.</span>`;
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
        not readable as "what this variable is worth". Switching off the
        redundant ones gives weights that mean what they look like.
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

function renderGroups() {
  const s = state.season;
  if (!s || s.status !== 'ready') return;
  const groups = {};
  for (const v of s.variables) (groups[v.group] ||= []).push(v);

  // Coefficients, keyed for lookup. Shown live so the effect of enabling a
  // variable is visible immediately -- including when it turns out to be ~0.
  const beta = {};
  if (state.fit) state.fit.keys.forEach((k, n) => { beta[k] = state.fit.beta[n]; });
  const maxAbs = Math.max(0.001, ...Object.values(beta).map(Math.abs));

  const opt = usingOptimized();
  document.getElementById('prebuilt').innerHTML = `
    <label class="vopt${opt ? ' active' : ''}">
      <input type="checkbox" ${opt ? 'checked' : ''} onchange="toggleVar('${OPTIMIZED}')">
      <span class="vopt-name">Pool optimized</span>
      <span class="v-tag">validated</span>
      <span class="vopt-sub">Built by the backtested pipeline, not fitted here</span>
    </label>
    <span class="vopt-or">or fit your own from</span>`;

  document.getElementById('groups').innerHTML = Object.entries(groups).map(([g, vars]) => `
    <div class="group">
      <p class="g-name">${g}</p>
      ${vars.map(v => {
        const on = state.enabled.has(v.key);
        const b = beta[v.key];
        return `
        <label class="v${on ? ' active' : ''}">
          <input type="checkbox" ${on ? 'checked' : ''} onchange="toggleVar('${v.key}')">
          <span class="v-label">${v.label}</span>
          ${coefHTML(b, maxAbs)}
        </label>`;
      }).join('')}
    </div>`).join('');
}

/* A coefficient is POINTS OF MARGIN per standard deviation of edge. The bar is
 * relative to the largest coefficient currently fitted, so the comparison is
 * between the variables actually in the model.
 *
 * WEAK_COEF is in those same points-per-SD units. It was 0.05 while this was a
 * logistic fit on log-odds; carried over unchanged into margin units it caught
 * 4% of fitted coefficients and the "no effect" marker was effectively dead
 * code. At 0.25 a variable has to move the predicted margin by less than half a
 * point across a full two-sigma swing in team quality to be called negligible,
 * which flags the bottom ~15% -- the band where a coefficient genuinely cannot
 * change a pick. */
const WEAK_COEF = 0.25;

function coefHTML(b, maxAbs) {
  if (b === undefined) return `<span class="coef off">—</span>`;
  const pct = Math.min(100, Math.abs(b) / maxAbs * 100);
  const weak = Math.abs(b) < WEAK_COEF;
  return `
    <span class="coef${b < 0 ? ' neg' : ''}${weak ? ' weak' : ''}" title="${weak ? 'Essentially no effect' : 'Points of margin per standard deviation'}">
      <span class="coef-bar"><i style="width:${pct}%"></i></span>
      <span class="coef-n">${b >= 0 ? '+' : ''}${b.toFixed(2)}</span>
    </span>`;
}

function toggleVar(key) {
  if (key === OPTIMIZED) {
    // Picking the prebuilt bracket replaces the fitted one entirely.
    state.enabled.clear();
    state.enabled.add(OPTIMIZED);
  } else {
    state.enabled.delete(OPTIMIZED);
    if (state.enabled.has(key)) state.enabled.delete(key);
    else state.enabled.add(key);
  }
  refit();
  renderGroups();
  render();
  updateHint();
}

function clearWeights() {
  state.enabled.clear();
  state.enabled.add(OPTIMIZED);
  refit();
  renderGroups();
  render();
  updateHint();
}

function updateHint() {
  const n = state.enabled.size;
  const el = document.getElementById('w-hint');
  if (!n) { el.textContent = ''; return; }
  const f = state.fit;
  el.textContent = f && f.oos
    ? `${n} on · ${(f.oos.accuracy * 100).toFixed(1)}% out-of-sample`
    : `${n} on`;
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
        const on = state.enabled.has(v.key);
        return `
        <div class="d-row${on ? ' lit' : ''}">
          <span class="d-lab">${v.label}</span>
          <span class="d-track"><i style="left:${pct}%"></i></span>
          <span class="d-val">${raw === null || raw === undefined ? '—' : fmt(raw)}</span>
          <button class="d-w" onclick="toggleVar('${v.key}')" title="${on ? 'Remove from the model' : 'Add to the model'}">${on ? '✓' : '+'}</button>
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
  renderGroups();
  render();
  updateHint();
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

  const priorEl = document.getElementById('prior-w');
  if (priorEl) {
    priorEl.addEventListener('input', e => {
      state.priorWeight = Number(e.target.value) / 100;
      document.getElementById('prior-v').textContent = `${e.target.value}%`;
      render();   // the blend changes picks, so the whole board is restated
    });
  }
  document.getElementById('clear-w').addEventListener('click', clearWeights);
  document.getElementById('d-close').addEventListener('click', closeDrawer);
  document.getElementById('scrim').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  await setYear(state.year);
}

// Track which team the drawer is showing so a weight change can refresh it.
const _openTeam = openTeam;
openTeam = function (i) { state.openIdx = i; _openTeam(i); };

init();
