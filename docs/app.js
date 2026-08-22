/* Bracket Lab — one page, two ways to fill a bracket.
 *
 *   Optimized   the bracket chosen in Python by the LOYO-validated method.
 *               Shipped precomputed in the season payload; the browser renders
 *               it and does not re-derive it.
 *
 *   Fit         the user switches variables on and off; a logistic regression
 *               is fitted live on real tournament games and its coefficients
 *               decide every matchup. Nobody has to guess a weight or a sign --
 *               the data supplies both, and they are shown.
 *
 * The fit excludes the displayed season (leave-one-year-out), so the
 * coefficients were never derived from the games being predicted.
 */

const ROUNDS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];

const state = {
  year: 2026,
  mode: 'pool',
  enabled: new Set(),   // variable keys switched on
  fit: null,            // {beta, n, converged}
  training: null,
  season: null,
  cache: {},
};

/* ---------- data ---------- */

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
  const keys = [...state.enabled];
  if (!state.training || !keys.length) { state.fit = null; return; }
  const cols = keys.map(k => state.training.keys.indexOf(k)).filter(i => i >= 0);
  const f = fitLogistic(state.training.games, cols, state.year);
  f.keys = keys;
  f.quality = fitQuality(state.training.games, cols, state.year, f.beta);
  // The honest number: fit on prior seasons, scored on seasons never seen.
  f.oos = crossValidate(state.training.games, cols, state.training.years, 2014);
  state.fit = f;
}

/* P(team a beats team b) under the fitted coefficients.
 *
 * Antisymmetric by construction: swapping a and b negates the differential and
 * so flips the probability exactly. */
function winProb(a, b) {
  const z = state.season.z, f = state.fit;
  let t = 0;
  for (let j = 0; j < f.keys.length; j++) {
    const col = z[f.keys[j]];
    if (col) t += f.beta[j] * ((col[a] || 0) - (col[b] || 0));
  }
  return sigmoid(t);
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

function anyEnabled() {
  return state.enabled.size > 0 && state.fit && state.fit.keys.length > 0;
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
    note.innerHTML = '';
    empty.hidden = false;
    empty.innerHTML = `
      <p class="e-title">${s ? s.message : 'Season unavailable.'}</p>
      <p class="e-sub">${s ? s.detail : ''}</p>`;
    return;
  }

  empty.hidden = true;
  weights.hidden = state.mode !== 'vars';

  if (state.mode === 'pool') {
    note.innerHTML = `<span class="tag">LOYO validated</span><span>${s.pool_optimized_note}</span>`;
  } else if (!anyEnabled()) {
    note.innerHTML = `<span class="tag alt">Pick variables</span><span>Switch on any variables above. A model is fitted to real tournament games and its coefficients decide every matchup.</span>`;
  } else {
    const f = state.fit, o = f.oos;
    // Lead with out-of-sample. In-sample is shown second and labelled, because
    // it always looks better and always will.
    note.innerHTML = `<span class="tag alt">Fitted</span><span>` +
      `${f.keys.length} variable${f.keys.length > 1 ? 's' : ''}, fitted on ${f.n.toLocaleString()} games from seasons before ${state.year}. ` +
      (o ? `<strong>${(o.accuracy * 100).toFixed(1)}% out-of-sample</strong> across ${o.seasons} held-out seasons ` +
           `(${(f.quality.accuracy * 100).toFixed(1)}% in-sample).`
         : `Not enough history to test out-of-sample.`) +
      `</span>`;
  }

  document.getElementById('equation').innerHTML =
    state.mode === 'vars' && anyEnabled() ? equationHTML() : '';

  const rounds = state.mode === 'pool' || !anyEnabled() ? solveFromPicks() : solveByFit();
  board.innerHTML = rounds.map((games, r) => `
    <div class="round" style="--n:${games.length}">
      <p class="r-label">${ROUNDS[r]}</p>
      ${games.map(g => gameHTML(g, r)).join('')}
    </div>`).join('');
}

/* The fitted model, written out.
 *
 * Terms are ordered by magnitude rather than by menu position, so the variables
 * actually carrying the model come first. Every delta is a difference in
 * standard deviations between the two teams, which is why a coefficient reads as
 * log-odds per standard deviation of edge.
 */
function equationHTML() {
  const f = state.fit;
  const label = Object.fromEntries(state.season.variables.map(v => [v.key, v.label]));

  const terms = f.keys
    .map((k, i) => ({ k, b: f.beta[i], label: label[k] || k }))
    .sort((a, b) => Math.abs(b.b) - Math.abs(a.b));

  const body = terms.map((t, i) => {
    const sign = t.b < 0 ? '\u2212' : '+';
    const mag = Math.abs(t.b).toFixed(2);
    const weak = Math.abs(t.b) < 0.05;
    return `<span class="term${weak ? ' weak' : ''}" title="${weak ? 'Essentially no contribution' : ''}">` +
           `${i === 0 && t.b >= 0 ? '' : `<i class="op">${sign}</i>`}` +
           `<b>${mag}</b><span class="dv">\u0394${t.label}</span></span>`;
  }).join('');

  return `
    <div class="eq">
      <p class="eq-head">
        <span class="eq-lhs">log-odds(team A beats team B)</span>
        <span class="eq-eq">=</span>
      </p>
      <p class="eq-body">${body}</p>
      <p class="eq-foot">
        \u0394 is team A minus team B, in standard deviations within the season.
        No intercept: swapping the teams flips the sign exactly.
      </p>
    </div>`;
}

/* Three separate signals, deliberately not conflated:
 *
 *   PICKED   this bracket advanced the team. The primary state.
 *   UPSET    the pick is the lower-seeded team. A property of the pick.
 *   WRONG    the team did not actually win. A property of reality.
 *
 * They were previously collapsed: upsets were amber, which read as "wrong" even
 * though the board showed no real outcomes at all. Now a pick that lost is
 * struck through and the team that actually won is marked, so "bold choice" and
 * "missed" can never be mistaken for each other.
 */
function gameHTML(g, round) {
  const pa = g.p === undefined ? null : g.p;
  const truth = state.season.actual ? new Set(state.season.actual[round]) : null;
  // A game is only gradeable if reality reached it: both teams must be ones the
  // real tournament actually put in this game.
  const gradeable = truth && (truth.has(g.a) || truth.has(g.b));
  return `
    <div class="game">
      ${sideHTML(g.a, g.win === g.a, pa === null ? null : pa, g.b, gradeable ? truth.has(g.a) : null)}
      ${sideHTML(g.b, g.win === g.b, pa === null ? null : 1 - pa, g.a, gradeable ? truth.has(g.b) : null)}
    </div>`;
}

function sideHTML(i, picked, p, oppI, actuallyWon) {
  const t = state.season.teams[i];
  const upset = picked && state.season.teams[oppI].seed < t.seed;
  const wrong = picked && actuallyWon === false;
  const missed = !picked && actuallyWon === true;   // the team that really won
  return `
    <button class="side${picked ? ' picked' : ''}${upset ? ' upset' : ''}` +
    `${wrong ? ' wrong' : ''}${missed ? ' missed' : ''}" onclick="openTeam(${i})">
      <span class="seed">${t.seed}</span>
      <span class="tname">${t.name}</span>
      ${upset ? '<span class="badge up" title="Lower seed picked">UPSET</span>' : ''}
      ${missed ? '<span class="badge won" title="This team actually won">WON</span>' : ''}
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

/* A coefficient is log-odds per standard deviation of edge. The bar is relative
 * to the largest coefficient currently fitted, so the comparison is between the
 * variables actually in the model. */
function coefHTML(b, maxAbs) {
  if (b === undefined) return `<span class="coef off">—</span>`;
  const pct = Math.min(100, Math.abs(b) / maxAbs * 100);
  const weak = Math.abs(b) < 0.05;
  return `
    <span class="coef${b < 0 ? ' neg' : ''}${weak ? ' weak' : ''}" title="${weak ? 'Essentially no effect' : 'Log-odds per standard deviation'}">
      <span class="coef-bar"><i style="width:${pct}%"></i></span>
      <span class="coef-n">${b >= 0 ? '+' : ''}${b.toFixed(2)}</span>
    </span>`;
}

function toggleVar(key) {
  if (state.enabled.has(key)) state.enabled.delete(key);
  else state.enabled.add(key);
  if (state.mode !== 'vars') setMode('vars');
  refit();
  renderGroups();
  render();
  updateHint();
}

function clearWeights() {
  state.enabled.clear();
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

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.mode').forEach(b => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', String(on));
  });
  render();
}

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
  ]);
  document.getElementById('years').innerHTML = idx.seasons.map(s => `
    <button class="yr${s.year === state.year ? ' on' : ''}" data-year="${s.year}"
            onclick="setYear(${s.year})">${s.year}</button>`).join('');

  document.querySelectorAll('.mode').forEach(b =>
    b.addEventListener('click', () => setMode(b.dataset.mode)));
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
