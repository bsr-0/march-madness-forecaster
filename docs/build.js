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

async function initBuild() {
  try {
    // Cache-buster tracks the schema, so a contract bump can never be served
    // a stale artifact from cache.
    const res = await fetch(`data/candidates_2026.json?v=${EXPECTED_ARTIFACT_SCHEMA}`);
    if (!res.ok) throw new Error(res.status);
    const loaded = await res.json();
    // Refuse an artifact this code does not understand rather than rendering
    // something plausible from it. The contract is owned by
    // src/product/artifact_contract.py and mirrored in selection.js.
    validateArtifact(loaded);
    ARTIFACT = loaded;
  } catch (e) {
    document.getElementById('build-context').textContent =
      'Bracket data is unavailable right now.';
    console.error('artifact rejected:', e.message);
    return;
  }
  renderStrategyCards();
  generateBrackets();
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
  generateBrackets();
}

/* PREFERENCE CONTROLS ARE DELIBERATELY ABSENT FROM V1.
 *
 * The engine supports the frozen preference predicates and they remain
 * canonical and parity-tested in selection.js / src/product/selection.py. They
 * are not exposed here because seed-shaped controls ("at least two 2/3 seeds in
 * the Final Four") are implementation vocabulary: they read as configuring an
 * optimizer rather than choosing how to play.
 *
 * The intended surface is an "angle" -- a basketball philosophy such as
 * rebounding or three-point shooting -- and no angle may ship until research
 * establishes that it produces materially different brackets, retains
 * acceptable expected score, is stable across seasons, is computable from
 * pre-tournament information, and can be frozen and versioned.
 *
 * Until then the product offers two objectives and returns two brackets rather
 * than manufacturing a third.
 */

function generateBrackets() {
  if (!ARTIFACT) return;
  const sel = selectWithAlternative(ARTIFACT, buildState.objective);
  buildState.selected = sel;

  const results = document.getElementById('build-results');
  if (!sel.length) {
    results.style.display = '';
    document.getElementById('result-cards').innerHTML =
      `<p class="empty-note">No brackets match that combination.</p>`;
    document.getElementById('compare-wrap').innerHTML = '';
    return;
  }

  /* Names state why each bracket exists. Never 'Bracket 1 / 2 / 3'.
   *
   * The second slot is filled only when selectWithAlternative found a materially
   * different bracket that keeps most of the objective's value. When it did not,
   * one bracket is shown -- the product does not invent a second strategy to
   * fill a slot. */
  const ROLE = buildState.objective === 'ev'
    ? ['Model Favorite', 'Alternative']
    : ['Pool Upside', 'Alternative'];
  const ROLE_NOTE = buildState.objective === 'ev'
    ? ['The highest expected-score bracket.',
       'A different bracket that keeps most of the expected value.']
    : ['The bracket with the best chance of finishing first.',
       'A different route to the same goal.'];
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
      <p class="rc-role-note">${ROLE_NOTE[n]}</p>
      <p class="rc-champ">${prettyName(s.champion_id)} <span class="rc-seed">(${s.champion_seed})</span></p>
      <p class="rc-f4">Final Four: ${s.final_four.map(t => `${prettyName(t.id)} (${t.seed})`).join(' · ')}</p>
      <p class="rc-meter"><span>Expected points</span><span class="rc-dots">${dots(s.ev, evs)}</span></p>
      <p class="rc-meter"><span>Pool upside</span><span class="rc-dots">${dots(s.p1, p1s)}</span></p>
      <p class="rc-dd">${s.double_digit_s16} double-digit seed(s) in the Sweet 16</p>
      ${n > 0 ? `<ul class="rc-why">${prettifyReasons(whyThisDiffers(ARTIFACT, sel[n], sel[0])).map(r => `<li>${r}</li>`).join('')}</ul>` : ''}
      <button class="rc-view" onclick="viewGeneratedBracket(${n})">View bracket</button>
    </div>`).join('');

  const m = ARTIFACT.meta || {};
  /* The replay disclosure is not optional copy. The ${m.year} tournament has
   * already been played, and these brackets are a replay against it — never a
   * live forecast and never evidence of predictive accuracy. */
  document.getElementById('compare-wrap').innerHTML = `
    <p class="compare-note">
      Pool upside assumes a ${m.p1_pool_size || 30}-opponent pool with typical public picks.
      It is not a universal probability of winning any pool.
    </p>
    <p class="compare-note">
      This is a replay of the ${m.year || 2026} field, shown so you can see how the
      builder behaves on a real bracket. It is not a live forecast.
    </p>`;
  results.style.display = '';
}

/* Canonical display name for a team id.
 *
 * Reads the artifact's `teams[].name` rather than transforming the slug. Slugs
 * do not round-trip -- `texas_a_m` is "Texas A&M", not "Texas A M" -- so any
 * client-side derivation is wrong for a predictable set of schools.
 */
function prettyName(id) {
  const t = ARTIFACT && ARTIFACT.teams.find(x => x.id === id);
  return t ? t.name : id;
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
  const s = candidateSummary(ARTIFACT, idx);
  window.GENERATED_ROUNDS = candidateToRounds(ARTIFACT, idx, mkTeam);

  const role = (buildState.objective === 'ev'
    ? ['Model Favorite', 'Alternative']
    : ['Pool Upside', 'Alternative'])[n];
  document.getElementById('detail-title').textContent =
    `${role} · ${prettyName(s.champion_id)} to win it`;

  setActiveTab('bracket');
  showGeneratedBracket();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBuild);
} else {
  initBuild();
}
