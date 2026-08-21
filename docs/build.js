/* "Build My Bracket" — the product flow.
 *
 * Presentation only. Every selection decision comes from selection.js, which is
 * a mirror of the canonical Python in src/product/selection.py. This file must
 * not filter, rank or choose brackets itself.
 *
 * The three strategies are the frozen v1 set. There is no Balanced blend and no
 * Contrarian ownership penalty — neither has been defined or measured, so
 * neither ships.
 */

let ARTIFACT = null;
let buildState = { objective: 'ev', preference: 'none', teamId: null, selected: [] };

/* Frozen v1 strategies. "Your Preference" is not a third objective — it is one
 * of the two measured objectives applied inside a user-chosen constraint, which
 * is more honest than inventing a formula that has never been evaluated. */
const BUILD_STRATEGIES = [
  { key: 'ev', icon: '\u{1F9E0}', label: 'Trust the Model',
    blurb: 'Highest expected points.',
    detail: 'The bracket our simulations score highest on average.' },
  { key: 'p1', icon: '\u{1F3AF}', label: 'Win My Pool',
    blurb: 'Best chance of finishing first.',
    detail: 'Takes positions designed to beat a field of other brackets, not to maximise points.' },
];

const PREF_GROUPS = [
  { title: 'Final Four', key: 'f4', options: [
      { id: 'none', label: 'No preference' },
      { id: 'f4_mostly_favorites', label: 'Mostly favorites' },
      { id: 'f4_at_least_1_two_three', label: 'At least one 2 or 3 seed' },
      { id: 'f4_at_least_2_two_three', label: 'At least two 2 or 3 seeds' },
  ]},
  { title: 'Cinderellas', key: 's16', options: [
      { id: 'none', label: 'No preference' },
      { id: 's16_no_double_digit', label: 'Chalk — favorites advance' },
      { id: 's16_at_least_1_double_digit', label: 'One double-digit seed in the Sweet 16' },
      { id: 's16_at_least_2_double_digit', label: 'Two or more' },
  ]},
];

async function initBuild() {
  try {
    const res = await fetch('data/candidates_2026.json?v=1');
    if (!res.ok) throw new Error(res.status);
    ARTIFACT = await res.json();
  } catch (e) {
    document.getElementById('build-context').textContent =
      'Bracket data is unavailable right now.';
    return;
  }
  const m = ARTIFACT.meta || {};
  document.getElementById('build-context').textContent =
    `${ARTIFACT.year} replay · ${ARTIFACT.candidates.length.toLocaleString()} candidate brackets ` +
    `from ${(m.n_sims || 0).toLocaleString()} simulated tournaments.`;
  renderStrategyCards();
  renderPrefGrid();
}

function renderStrategyCards() {
  document.getElementById('strategy-cards').innerHTML = BUILD_STRATEGIES.map(s => `
    <button class="strategy-card${buildState.objective === s.key ? ' selected' : ''}"
            onclick="setObjective('${s.key}')">
      <span class="sc-icon">${s.icon}</span>
      <span class="sc-label">${s.label}</span>
      <span class="sc-blurb">${s.blurb}</span>
      <span class="sc-detail">${s.detail}</span>
    </button>`).join('');
}

function setObjective(key) {
  buildState.objective = key;
  renderStrategyCards();
  if (buildState.selected.length) generateBrackets();
}

/* Feasibility text comes from the artifact's full-bank frequencies, never from
 * counting candidates — the candidate list deliberately over-samples unlikely
 * champions and is not a probability sample. */
function freqLabel(prefId) {
  if (prefId === 'none') return '';
  const f = constraintFrequency(ARTIFACT, prefId);
  if (f == null) return '';
  return `happens in about ${Math.round(f * 10)} of 10 simulated tournaments`;
}

function renderPrefGrid() {
  document.getElementById('pref-grid').innerHTML = PREF_GROUPS.map(g => `
    <div class="pref-group">
      <p class="pref-title">${g.title}</p>
      ${g.options.map(o => {
        const active = buildState.preference === o.id ||
          (o.id === 'none' && !g.options.some(x => x.id !== 'none' && x.id === buildState.preference));
        const isOn = buildState.preference === o.id;
        return `<label class="pref-opt${isOn ? ' on' : ''}">
          <input type="radio" name="pref-${g.key}" ${isOn ? 'checked' : ''}
                 onchange="setPreference('${o.id}','${g.key}')">
          <span class="pref-label">${o.label}</span>
          <span class="pref-freq">${freqLabel(o.id)}</span>
        </label>`;
      }).join('')}
    </div>`).join('');
}

/* One preference at a time in v1. Combining predicates is supported by the
 * engine but multiplies the rare-combination cases, and the feasibility warning
 * story is not designed yet. */
function setPreference(id, group) {
  buildState.preference = id;
  renderPrefGrid();
  if (buildState.selected.length) generateBrackets();
}

function generateBrackets() {
  if (!ARTIFACT) return;
  const sel = selectBrackets(ARTIFACT, buildState.objective, buildState.preference, null, 3);
  buildState.selected = sel;

  const results = document.getElementById('build-results');
  if (!sel.length) {
    results.style.display = '';
    document.getElementById('result-cards').innerHTML =
      `<p class="empty-note">No brackets match that combination.</p>`;
    document.getElementById('compare-wrap').innerHTML = '';
    return;
  }

  const ROLE = ['Model Favorite', 'Alternative', 'Third Option'];
  const summaries = sel.map(i => candidateSummary(ARTIFACT, i));
  const evs = ARTIFACT.candidates.map(c => c.ev);
  const p1s = ARTIFACT.candidates.map(c => c.p1);
  const dots = (v, arr) => {
    const lo = Math.min(...arr), hi = Math.max(...arr);
    const n = hi > lo ? Math.round(1 + 4 * (v - lo) / (hi - lo)) : 3;
    return '●'.repeat(n) + '○'.repeat(5 - n);
  };

  document.getElementById('result-cards').innerHTML = summaries.map((s, n) => `
    <div class="result-card">
      <p class="rc-role">${ROLE[n]}</p>
      <p class="rc-champ">${prettyName(s.champion_id)} <span class="rc-seed">(${s.champion_seed})</span></p>
      <p class="rc-f4">Final Four: ${s.final_four.map(t => `${prettyName(t.id)} (${t.seed})`).join(' · ')}</p>
      <p class="rc-meter"><span>Expected points</span><span class="rc-dots">${dots(s.ev, evs)}</span></p>
      <p class="rc-meter"><span>Pool upside</span><span class="rc-dots">${dots(s.p1, p1s)}</span></p>
      <p class="rc-dd">${s.double_digit_s16} double-digit seed(s) in the Sweet 16</p>
      ${n > 0 ? `<ul class="rc-why">${prettifyReasons(whyThisDiffers(ARTIFACT, sel[n], sel[0])).map(r => `<li>${r}</li>`).join('')}</ul>` : ''}
      <button class="rc-view" onclick="viewGeneratedBracket(${n})">View bracket</button>
    </div>`).join('');

  const m = ARTIFACT.meta || {};
  document.getElementById('compare-wrap').innerHTML = `
    <p class="compare-note">
      Pool upside assumes a ${m.p1_pool_size || 30}-opponent pool with typical public picks.
      It is not a universal probability of winning any pool.
    </p>`;
  results.style.display = '';
}

function prettyName(id) {
  return id.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

/* Prettify team ids inside explanation strings.
 *
 * Done here rather than in selection.js: that file is a mirror of the canonical
 * Python and must stay semantically identical to it. Display formatting is the
 * browser's concern, so it belongs in the presentation layer. */
function prettifyReasons(reasons) {
  const ids = ARTIFACT.teams.map(t => t.id).sort((a, b) => b.length - a.length);
  return reasons.map(r => {
    let out = r;
    for (const id of ids) if (out.includes(id)) out = out.split(id).join(prettyName(id));
    return out;
  });
}

/* Hand the chosen bracket to the EXISTING renderer rather than building a second
 * bracket representation. */
function viewGeneratedBracket(n) {
  const idx = buildState.selected[n];
  const rounds = candidateToRounds(ARTIFACT, idx, mkTeam, log5);
  window.GENERATED_ROUNDS = rounds;
  delete roundsCache['generated'];
  setActiveTab('bracket');
  activateStrategy('generated');
  document.getElementById('tab-bracket').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBuild);
} else {
  initBuild();
}
