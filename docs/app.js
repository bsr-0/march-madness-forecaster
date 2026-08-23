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

const OPTIMIZED = '__optimized__';

const state = {
  year: 2026,
  enabled: new Set([OPTIMIZED]),   // variable keys, or OPTIMIZED
  fit: null,            // {beta, n, converged}
  training: null,
  season: null,
  cache: {},
  tab: 'bracket',       // 'bracket' | 'diagnostics'
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
 * spread is what carries the margin into a probability. */
function winProb(a, b) {
  return winProbFromMargin(margin(a, b), state.fit.sigma);
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

  // The two tabs share one variable selection, so any change to it has to reach
  // whichever one is showing.
  if (state.tab === 'diagnostics') renderDiagnostics();

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
  weights.hidden = false;   // the panel is always the control surface now

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
           `— against ${(f.quality.accuracy * 100).toFixed(1)}% on the games it was fitted to.`
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
 * log-odds per standard deviation of edge.
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

/* ---------- diagnostics ---------- */

/* A workbench, not product surface. Everything here refits by ordinary least
 * squares; the board keeps using the ridge fit. Where the two disagree, the
 * ridge was doing real work, and saying so is the point.
 */

const P_FMT = p =>
  p === null ? '—' : p < 1e-4 ? '&lt;0.0001' : p.toFixed(4);

function fmt(v, d = 2) {
  return v === null || v === undefined || !isFinite(v) ? '—' : v.toFixed(d);
}

function renderDiagnostics() {
  const el = document.getElementById('diagnostics');
  const keys = state.fit ? state.fit.keys : [];

  if (!state.training || !keys.length) {
    el.innerHTML = `
      <div class="d-empty">
        <p class="e-title">No model to diagnose.</p>
        <p class="e-sub">Switch on one or more variables on the Bracket tab. The
        prebuilt bracket is not a regression, so there is nothing to test.</p>
      </div>`;
    return;
  }

  const cols = keys.map(k => state.training.keys.indexOf(k)).filter(i => i >= 0);
  const r = diagnose(state.training.games, cols, keys, state.year);
  if (r.error) {
    el.innerHTML = `<div class="d-empty"><p class="e-title">${r.error}</p></div>`;
    return;
  }

  el.innerHTML = [
    diagPreamble(r),
    diagSignificance(r),
    diagVIF(r),
    diagSigns(r),
    diagResiduals(r),
    diagIntercept(r),
    diagBaselines(r),
    diagNext(r),
  ].join('');
}

function diagPreamble(r) {
  return `
    <div class="d-head">
      <p class="d-h1">Regression diagnostics</p>
      <p class="d-sub2">
        Ordinary least squares on ${r.fit.n.toLocaleString()} tournament games from seasons
        before ${state.year}, ${r.fit.k} variable${r.fit.k > 1 ? 's' : ''}, ${r.fit.df.toLocaleString()} degrees of freedom.
        Residual spread ${fmt(Math.sqrt(r.fit.sigma2))} points.
      </p>
      <p class="d-warn">
        <b>Read these as screening, not proof.</b> Two reasons. The board's model is
        ridge-penalised and a penalised coefficient has no standard error in the ordinary
        sense, so this refits <i>without</i> the penalty — the numbers below will not match
        the board's equation exactly. And you chose these variables after seeing results,
        which makes every p-value here optimistic in a way no correction on this page
        undoes. Fold-to-fold coefficient stability, on the Bracket tab, is the more honest
        signal.
      </p>
    </div>`;
}

function diagSignificance(r) {
  const rows = r.coefficients.map(c => `
    <tr class="${c.significant ? '' : 'dim'}">
      <td class="k">${c.key}</td>
      <td class="n">${fmt(c.beta)}</td>
      <td class="n">${fmt(c.se)}</td>
      <td class="n">${fmt(c.t, 2)}</td>
      <td class="n ${c.significant ? 'sig' : ''}">${P_FMT(c.p)}</td>
      <td class="n ci">${fmt(c.ci[0])} to ${fmt(c.ci[1])}</td>
    </tr>`).join('');
  const nSig = r.coefficients.filter(c => c.significant).length;
  return `
    <div class="d-card">
      <p class="d-h2">1 · Significance</p>
      <p class="d-note">
        ${nSig} of ${r.coefficients.length} coefficients are distinguishable from zero at
        the 5% level. A coefficient whose interval straddles zero is not evidence that the
        variable does nothing — with collinear inputs it usually means the credit could not
        be assigned between overlapping columns.
      </p>
      <table class="d-table">
        <thead><tr><th>variable</th><th class="n">β</th><th class="n">std. error</th>
        <th class="n">t</th><th class="n">p</th><th class="n">95% interval</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="d-foot">β is points of scoring margin per standard deviation of edge.</p>
    </div>`;
}

function diagVIF(r) {
  const rows = r.vif.map(v => `
    <tr>
      <td class="k">${v.key}</td>
      <td class="n"><span class="pill ${v.severity}">${fmt(v.vif)}</span></td>
      <td class="n">${fmt(v.seInflation)}&times;</td>
      <td class="sev ${v.severity}">${v.severity}</td>
    </tr>`).join('');
  const worst = r.vif.reduce((a, b) => (b.vif > a.vif ? b : a), r.vif[0]);
  return `
    <div class="d-card">
      <p class="d-h2">2 · Multicollinearity</p>
      <p class="d-note">
        Variance inflation measures how much of a variable is already explained by the
        others. 1 is orthogonal; above 5 the coefficient is being estimated from a thin
        residual slice of the column; above 10 is conventionally severe.
        Worst here: <b>${worst.key}</b> at ${fmt(worst.vif)}, which widens its confidence
        interval by ${fmt(worst.seInflation)}&times; against an uncorrelated column.
      </p>
      <table class="d-table">
        <thead><tr><th>variable</th><th class="n">VIF</th><th class="n">interval widened</th><th>severity</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="d-foot">
        Measured about zero rather than about the column mean, because this model has no
        intercept to absorb the means — the usual centred formula would report the wrong
        number here.
      </p>
    </div>`;
}

function diagSigns(r) {
  const rows = r.signs.map(s => {
    const flag = s.implausible ? 'implausible' : s.wrongSign ? 'wrong sign' : s.flipped ? 'flipped' : 'ok';
    return `
    <tr class="${flag === 'ok' ? '' : 'flag'}">
      <td class="k">${s.key}</td>
      <td class="n">${fmt(s.alone)}</td>
      <td class="n">${fmt(s.joint)}</td>
      <td class="sev ${flag === 'ok' ? 'low' : flag === 'flipped' ? 'moderate' : 'severe'}">${flag}</td>
    </tr>`;
  }).join('');
  const bad = r.signs.filter(s => s.wrongSign || s.implausible || s.flipped).length;
  return `
    <div class="d-card">
      <p class="d-h2">3 · Signs and magnitudes</p>
      <p class="d-note">
        Every variable is standardised upstream so that <b>higher is better</b>. A negative
        coefficient therefore contradicts the construction of its own column and needs an
        explanation. The usual one is collinearity, not a reversed effect — so the
        coefficient fitted <i>alone</i> is shown beside the joint one. A variable positive
        alone and negative in company has not changed what it does; it has been assigned
        someone else's credit.
        ${bad ? `<b>${bad}</b> flagged below.` : 'Nothing flagged.'}
      </p>
      <table class="d-table">
        <thead><tr><th>variable</th><th class="n">β alone</th><th class="n">β jointly</th><th>verdict</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="d-foot">
        A magnitude beyond ±15 points per standard deviation is marked implausible: that is
        larger than the spread of the thing being predicted, and is one side of a cancelling
        pair rather than an effect.
      </p>
    </div>`;
}

function residualPlot(res) {
  const W = 620, H = 240, PAD = 34;
  const xs = res.points.map(p => p[0]), ys = res.points.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yMax = Math.max(...ys.map(Math.abs));
  const sx = v => PAD + ((v - x0) / (x1 - x0 || 1)) * (W - PAD - 10);
  const sy = v => H / 2 - (v / (yMax || 1)) * (H / 2 - 12);

  const dots = res.points
    .map(([f, e]) => `<circle cx="${sx(f).toFixed(1)}" cy="${sy(e).toFixed(1)}" r="1.6"/>`)
    .join('');

  // Band means, drawn as a line: a run of same-signed points is curvature the
  // linear form is not capturing.
  const path = res.bands.filter(b => b.n > 3)
    .map((b, i) => `${i ? 'L' : 'M'}${sx((b.lo + b.hi) / 2).toFixed(1)},${sy(b.meanResid).toFixed(1)}`)
    .join(' ');

  const sigLine = s => `<line x1="${PAD}" y1="${sy(s).toFixed(1)}" x2="${W - 10}" y2="${sy(s).toFixed(1)}" class="sig-line"/>`;

  return `
    <svg class="resid" viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Residuals against fitted values">
      <line x1="${PAD}" y1="${H / 2}" x2="${W - 10}" y2="${H / 2}" class="axis"/>
      ${sigLine(2 * res.sigma)}${sigLine(-2 * res.sigma)}
      <g class="dots">${dots}</g>
      <path d="${path}" class="bandline"/>
      <text x="${PAD}" y="12" class="lab">residual (points)</text>
      <text x="${W - 10}" y="${H - 4}" class="lab end">fitted margin →</text>
      <text x="${PAD - 6}" y="${sy(2 * res.sigma) - 3}" class="lab">+2σ</text>
    </svg>`;
}

function diagResiduals(r) {
  const res = r.residuals;
  const het = res.heteroscedasticity;
  const outlierMult = res.outlierRate / res.expectedOutlierRate;
  const bandRuns = res.bands.filter(b => b.n > 3).map(b => (b.meanResid > 0 ? '+' : '−')).join('');
  return `
    <div class="d-card">
      <p class="d-h2">4 · Residuals</p>
      ${residualPlot(res)}
      <div class="d-grid">
        <div><span class="lbl">Mean residual</span><span class="val">${fmt(res.meanResid, 3)}</span>
          <span class="sub">≈0 by construction of least squares</span></div>
        <div><span class="lbl">Heteroscedasticity</span>
          <span class="val ${het.significant ? 'bad' : 'good'}">${het.significant ? 'present' : 'not detected'}</span>
          <span class="sub">Breusch–Pagan χ²=${fmt(het.statistic)}, p=${P_FMT(het.p)}${
            het.significant ? `; error ${het.widensWithMargin ? 'grows' : 'shrinks'} with |predicted margin|` : ''}</span></div>
        <div><span class="lbl">Outliers beyond 3σ</span>
          <span class="val ${outlierMult > 2 ? 'bad' : 'good'}">${res.outlierCount}</span>
          <span class="sub">${(100 * res.outlierRate).toFixed(2)}% vs 0.27% expected — ${fmt(outlierMult, 1)}× normal</span></div>
        <div><span class="lbl">Band-mean signs</span><span class="val mono">${bandRuns}</span>
          <span class="sub">low → high fitted; a long run of one sign is curvature</span></div>
      </div>
      <p class="d-note">
        ${het.significant
          ? (het.widensWithMargin
            ? 'Error spread <b>grows</b> with the size of the predicted margin: the model is least reliable exactly where it is most emphatic. The board converts margin to probability with one σ for every game, so it is <b>over-confident about mismatches</b> and slightly under-confident about close games.'
            : 'Error spread <b>shrinks</b> as the predicted margin grows: the model is more reliable about mismatches than about close games. Since the board uses one σ everywhere, that makes it <b>under-confident about mismatches</b> — a 20-point favourite deserves a firmer probability than it is being given — and over-confident about coin-flips.')
          : 'Error spread does not vary detectably with the size of the prediction, so the single σ the board uses to convert margin into probability is a reasonable simplification.'}
      </p>
      <p class="d-foot">
        Heavy tails are expected here regardless: a tournament blowout is a real event no
        linear model in season averages can anticipate. The line traces mean residual by
        band — it should wander around zero, not arc.
      </p>
    </div>`;
}

function diagIntercept(r) {
  const it = r.intercept;
  if (!it) return '';
  return `
    <div class="d-card">
      <p class="d-h2">5 · The no-intercept assumption</p>
      <p class="d-note">
        Forcing the line through the origin is a real restriction, so here is what it costs.
        Fitting the same variables <i>with</i> a constant gives an intercept of
        <b>${fmt(it.intercept)}</b> points (std. error ${fmt(it.se)}, t=${fmt(it.t)},
        p=${P_FMT(it.p)}) — ${it.significant
          ? '<b class="bad">distinguishable from zero</b>'
          : '<b class="good">not distinguishable from zero</b>'}.
      </p>
      <div class="d-grid">
        <div><span class="lbl">RMSE without intercept</span><span class="val">${fmt(it.rmseWithout)}</span></div>
        <div><span class="lbl">RMSE with intercept</span><span class="val">${fmt(it.rmseWith)}</span>
          <span class="sub">${fmt(it.rmseWith - it.rmseWithout, 3)} difference</span></div>
        <div><span class="lbl">R² about zero</span><span class="val">${fmt(it.r2ZeroWithout, 3)}</span>
          <span class="sub">baseline: the teams are even</span></div>
        <div><span class="lbl">R² about the mean</span><span class="val">${fmt(it.r2MeanWithout, 3)}</span>
          <span class="sub">baseline: everyone wins by ${fmt(it.meanY, 1)}</span></div>
      </div>
      <p class="d-note">
        <b>Why the two R² figures differ so much.</b> Through the origin, R² is conventionally
        quoted against "predict zero". About the mean it is quoted against "predict
        ${fmt(it.meanY, 1)} points every time" — but that baseline is only available to
        someone who already knows which team to list first. Rows here are ordered
        better-seed-first, so the mean margin is positive and the mean-centred figure is
        scored against a baseline that has quietly been told the answer. The
        <b>about-zero</b> figure is the defensible one for this design, and it is what the
        board reports.
      </p>
      <p class="d-foot">
        The case for excluding the constant is structural, not statistical: rows are
        differentials, so swapping the two teams must negate the prediction exactly. A
        constant would assert that whoever is written first wins by ${fmt(it.intercept)}
        points before anyone looks at the teams — a fact about row order, not basketball.
        ${it.significant
          ? 'That it tests significant reflects the seed-first ordering, and is a reason to distrust the ordering rather than to add a constant.'
          : 'The test agreeing costs nothing and is reassuring.'}
      </p>
    </div>`;
}

function diagBaselines(r) {
  const rows = r.baselines.map((b, i) => `
    <tr class="${i === r.baselines.length - 1 ? 'best' : ''}">
      <td class="k">${b.name}<span class="sub2">${b.note}</span></td>
      <td class="n">${fmt(b.rmse)}</td>
      <td class="n">${fmt(b.mae)}</td>
      <td class="n">${fmt(b.r2, 3)}</td>
      <td class="n">${(100 * b.accuracy).toFixed(1)}%</td>
    </tr>`).join('');
  return `
    <div class="d-card">
      <p class="d-h2">6 · Against a naive baseline</p>
      <p class="d-note">
        None of the above means anything without something to beat. These are in-sample on
        the training seasons, so they are all flattered equally; the out-of-sample figure on
        the Bracket tab is the one that counts.
      </p>
      <table class="d-table">
        <thead><tr><th>model</th><th class="n">RMSE</th><th class="n">MAE</th>
        <th class="n">R² (about 0)</th><th class="n">picks right</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="d-foot">
        The constant-only row is the trap: it looks respectable on accuracy because rows are
        ordered better-seed-first, so "always favour the first team" is the seed baseline in
        disguise. Beating it on RMSE is the low bar this model has to clear.
      </p>
    </div>`;
}

function diagNext(r) {
  const het = r.residuals.heteroscedasticity.significant;
  const worstVif = Math.max(...r.vif.map(v => v.vif));
  const items = [];
  if (worstVif >= 10) {
    items.push(`<li><b>Collinearity is severe</b> (max VIF ${fmt(worstVif)}). Switching off one of
      the overlapping ratings will barely move predictions and will make the remaining
      coefficients readable. This is the single biggest obstacle to interpreting the fit.</li>`);
  }
  if (het) {
    const w = r.residuals.heteroscedasticity.widensWithMargin;
    items.push(`<li><b>Error spread varies with the size of the prediction</b> — it
      ${w ? 'grows' : 'shrinks'} as the predicted margin gets larger. One σ is used for every
      game, so the probabilities are ${w ? 'over' : 'under'}-confident about mismatches and
      ${w ? 'under' : 'over'}-confident about close games. Modelling σ as a function of
      |predicted margin| would fix the calibration without moving a single pick, since the
      winner is decided by the sign of the margin and not by σ.</li>`);
  }
  if (r.residuals.outlierRate > 2 * r.residuals.expectedOutlierRate) {
    items.push(`<li><b>Tails are heavier than normal</b>
      (${(100 * r.residuals.outlierRate).toFixed(2)}% beyond 3σ against 0.27% expected).
      Squared error chases blowouts; a robust loss would fit the typical game better at the
      cost of the extremes.</li>`);
  }
  items.push(`<li><b>Regularisation is already in the shipped model</b> — ridge, chosen for
    out-of-sample error rather than readability. Lasso would drop the redundant ratings
    outright instead of splitting credit between them, which is worth comparing.</li>`);
  items.push(`<li><b>Non-linearity looks mild</b> from the band means, so a linear form is not
    obviously the binding constraint. Interactions between pace and efficiency would be the
    first thing to test if that changes.</li>`);

  return `
    <div class="d-card next">
      <p class="d-h2">Where this points</p>
      <ul class="d-list">${items.join('')}</ul>
    </div>`;
}

/* ---------- controls ---------- */

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  const onBracket = tab === 'bracket';
  document.getElementById('board').hidden = !onBracket;
  document.getElementById('equation').hidden = !onBracket;
  document.getElementById('mode-note').hidden = !onBracket;
  document.getElementById('diagnostics').hidden = onBracket;
  if (!onBracket) renderDiagnostics();
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

  document.getElementById('tabs').addEventListener('click', e => {
    if (e.target.dataset.tab) setTab(e.target.dataset.tab);
  });
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
