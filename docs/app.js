/* Bracket Lab — one page, two ways to fill a bracket.
 *
 *   Optimized   the bracket chosen in Python by the LOYO-validated method.
 *               Shipped precomputed in the season payload; the browser renders
 *               it and does not re-derive it.
 *
 *   My weights  the user weights pre-tournament stats and every game is decided
 *               by the weighted sum. This IS computed in the browser, and that
 *               is deliberate: it is arithmetic over standardised values that
 *               already shipped, not a model. Nothing here estimates a
 *               probability or reimplements the tournament engine.
 *
 * Z-scores arrive sign-corrected from Python, so "more of this is better" is
 * already baked into the data. Weights are therefore 0..5 and never negative:
 * asking a user to decide the sign of a coefficient would be asking them to
 * re-derive something the payload already knows, and getting it wrong would
 * silently favour (say) the worst defences.
 *
 * One consequence worth knowing: with a SINGLE variable the magnitude does not
 * matter. Scaling one column by a positive constant cannot reorder it, so
 * weight 1 and weight 5 give the same bracket. Magnitude only starts to matter
 * once two or more variables are competing.
 */

const ROUNDS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];

const state = {
  year: 2026,
  mode: 'pool',
  weights: {},      // key -> 0..5 (never negative; see note below)
  season: null,
  cache: {},
};

/* ---------- data ---------- */

async function loadSeason(year) {
  if (state.cache[year]) return state.cache[year];
  const res = await fetch(`data/season_${year}.json?v=1`);
  if (!res.ok) throw new Error(`season ${year} unavailable`);
  const data = await res.json();
  state.cache[year] = data;
  return data;
}

/* ---------- bracket solving ---------- */

/* Score a team under the current weights. Absent weights mean zero, so an
 * untouched control contributes nothing rather than a hidden default. */
function score(idx) {
  const z = state.season.z;
  let total = 0;
  for (const key in state.weights) {
    const w = state.weights[key];
    if (!w) continue;
    const col = z[key];
    if (col) total += w * (col[idx] || 0);
  }
  return total;
}

/* Play the bracket out under the weights.
 *
 * Ties break toward the better seed, then by index, so the board is stable and
 * a user changing one weight does not see unrelated games flip. */
function solveByWeights() {
  const teams = state.season.teams;
  let current = state.season.first_round.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const games = [], next = [];
    for (let g = 0; g < current.length; g += 2) {
      const a = current[g], b = current[g + 1];
      const sa = score(a), sb = score(b);
      let win;
      if (sa !== sb) win = sa > sb ? a : b;
      else if (teams[a].seed !== teams[b].seed) win = teams[a].seed < teams[b].seed ? a : b;
      else win = Math.min(a, b);
      games.push({ a, b, win, sa, sb });
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

function anyWeight() {
  return Object.values(state.weights).some(v => v);
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
  } else if (!anyWeight()) {
    note.innerHTML = `<span class="tag alt">Pick a variable</span><span>Choose one above and it alone decides every game. Add more to blend them.</span>`;
  } else {
    note.innerHTML = `<span class="tag alt">Your weights</span><span>${describeWeights()}</span>`;
  }

  const rounds = state.mode === 'pool' || !anyWeight() ? solveFromPicks() : solveByWeights();
  board.innerHTML = rounds.map((games, r) => `
    <div class="round" style="--n:${games.length}">
      <p class="r-label">${ROUNDS[r]}</p>
      ${games.map(g => gameHTML(g)).join('')}
    </div>`).join('');
}

function describeWeights() {
  const on = Object.entries(state.weights).filter(([, v]) => v);
  const byKey = Object.fromEntries(state.season.variables.map(v => [v.key, v.label]));
  if (on.length === 1) {
    // Magnitude is inert for a single variable, so the copy does not mention it.
    return `Every game goes to the team with better <strong>${byKey[on[0][0]]}</strong>.`;
  }
  const parts = on.sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => byKey[k]);
  return `Blending <strong>${parts.join('</strong>, <strong>')}</strong>` +
         `${on.length > 3 ? ` and ${on.length - 3} more` : ''}, weighted.`;
}

function gameHTML(g) {
  const t = state.season.teams;
  return `
    <div class="game">
      ${sideHTML(g.a, g.win === g.a, g.sa, g.b)}
      ${sideHTML(g.b, g.win === g.b, g.sb, g.a)}
    </div>`;
}

function sideHTML(i, won, sc, oppI) {
  const t = state.season.teams[i];
  const upset = won && state.season.teams[oppI].seed < t.seed;
  return `
    <button class="side${won ? ' win' : ''}${upset ? ' upset' : ''}" onclick="openTeam(${i})">
      <span class="seed">${t.seed}</span>
      <span class="tname">${t.name}</span>
      ${sc === null ? '' : `<span class="sc">${sc > 0 ? '+' : ''}${sc.toFixed(2)}</span>`}
    </button>`;
}

/* ---------- weights ---------- */

function renderGroups() {
  const s = state.season;
  if (!s || s.status !== 'ready') return;
  const groups = {};
  for (const v of s.variables) (groups[v.group] ||= []).push(v);

  document.getElementById('groups').innerHTML = Object.entries(groups).map(([g, vars]) => `
    <div class="group">
      <p class="g-name">${g}</p>
      ${vars.map(v => {
        const w = state.weights[v.key] || 0;
        return `
        <label class="v${w ? ' active' : ''}">
          <span class="v-label">${v.label}</span>
          <input type="range" min="0" max="5" step="1" value="${w}"
                 oninput="setWeight('${v.key}', this.value)">
          <span class="v-w">${w || '·'}</span>
        </label>`;
      }).join('')}
    </div>`).join('');
}

function setWeight(key, value) {
  const v = Number(value);
  if (v === 0) delete state.weights[key];
  else state.weights[key] = v;
  renderGroups();
  render();
  updateHint();
}

function clearWeights() {
  state.weights = {};
  renderGroups();
  render();
  updateHint();
}

function updateHint() {
  const n = Object.keys(state.weights).length;
  document.getElementById('w-hint').textContent =
    n === 0 ? '' : n === 1 ? '1 variable deciding' : `${n} variables blended`;
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
        const w = state.weights[v.key] || 0;
        return `
        <div class="d-row${w ? ' lit' : ''}">
          <span class="d-lab">${v.label}</span>
          <span class="d-track"><i style="left:${pct}%"></i></span>
          <span class="d-val">${raw === null || raw === undefined ? '—' : fmt(raw)}</span>
          <button class="d-w" onclick="bump('${v.key}')" title="Weight this variable">${w || '+'}</button>
        </div>`;
      }).join('')}
    </div>`).join('');

  document.getElementById('drawer').hidden = false;
  document.getElementById('scrim').hidden = false;
}

/* Weighting a variable from the drawer switches to the weights mode, since that
 * is the only mode where a weight changes anything. */
function bump(key) {
  const cur = state.weights[key] || 0;
  const next = cur >= 5 ? 0 : cur + 1;
  if (next === 0) delete state.weights[key]; else state.weights[key] = next;
  if (state.mode !== 'vars') setMode('vars');
  renderGroups();
  render();
  updateHint();
  const openIdx = state.openIdx;
  if (openIdx !== undefined) openTeam(openIdx);
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
  renderGroups();
  render();
}

async function init() {
  const idx = await fetch('data/seasons.json?v=1').then(r => r.json());
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
