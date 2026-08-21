/* ══════════════════════════════════════════════════════════════════
 * March Madness Forecaster — Bracket Picker
 * ══════════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────────
// SHELL AND BRACKET RENDERER
//
// What lives here: page-tab routing, the bracket board, the champion
// path, and grading a bracket against the real result.
//
// What does NOT live here, deliberately:
//
//   * Bracket construction. Brackets are chosen in Python and arrive
//     via the candidate artifact. The client-side simulate() and the
//     per-strategy pick() functions were a second implementation of
//     the model and have been removed.
//   * Win probabilities. The browser reads them from the artifact's
//     pairwise table; the client-side log5() is gone.
//   * A strategy catalog. The old one named construction algorithms
//     ("Pool Optimizer", "Region Beam Search", "Exhaustive Search")
//     and carried hardcoded P(1st) badges drawn from a window that
//     included the replay year. Users choose a goal now, in Build.
//
// Anything a user reads about accuracy belongs in Track Record, whose
// figures are computed in Python with the replay year excluded.
// ──────────────────────────────────────────────────────────────────


// ── Display constants ──

const ROUND_SHORT = {
  'Round of 64':  'R64',
  'Round of 32':  'R32',
  'Sweet 16':     'S16',
  'Elite 8':      'E8',
  'Final Four':   'F4',
  'Championship': 'Champ',
};

const REGIONS = ['East', 'West', 'South', 'Midwest'];

// round_name (as used in a bracket JSON's rounds[]) → the key
// docs/data/actual_2026.json's results_by_round uses. Matches ROUND_SHORT
// except Championship → CHAMP (ESPN_SCORING's key, not "Champ").
const ROUND_TO_ACTUAL_KEY = { ...ROUND_SHORT, 'Championship': 'CHAMP' };


// ── App state ──

// approach (pool, exhaustive, stat).
let actualData         = null;  // real 2026 outcome, see docs/data/actual_2026.json — the
                                 // 2026 tournament already concluded, this is a replay
let currentRound  = 'Round of 64';

// Multi-year pre-tournament team stats table, see docs/data/team_stats_by_year.json


// ──────────────────────────────────────────────────────────────────
// BOOT
// ──────────────────────────────────────────────────────────────────

/* Boot.
 *
 * The three precomputed bracket_2026*.json files and team_profiles.json are no
 * longer fetched: they fed the per-algorithm strategy catalog and the
 * client-side rating lookups, both of which are gone. Build fetches the
 * candidate artifact, Track Record fetches its own payload, and the only shared
 * fetch left is the real 2026 result used to grade a bracket.
 */
document.addEventListener('DOMContentLoaded', async () => {
  // A missing result file degrades to "no grading panel" rather than blocking
  // the page.
  let actual;
  try {
    actual = await fetch('data/actual_2026.json?v=2026-08-16d').then(r => r.json());
  } catch (err) {
    actual = null;
  }
  actualData = actual;

  // Build owns the default destination and loads its own artifact; Explore is
  // initialised from that same artifact once it has arrived.
  setActiveTab('build');
  initRecord();
});

// ──────────────────────────────────────────────────────────────────
// PAGE TABS
//
// Top-level destinations: Build, Explore and Track Record, plus the bracket
// detail view Build opens. Client-side show/hide only — every panel's DOM is
// present from the start, so switching is instant and never re-fetches.
// ──────────────────────────────────────────────────────────────────

// Three destinations plus the bracket detail view, which is reached from Build
// rather than from the nav.
const PAGE_TABS = ['build', 'explore', 'record', 'bracket'];

function setActiveTab(tab) {
  PAGE_TABS.forEach(name => {
    const el = document.getElementById(`tab-${name}`);
    if (el) el.style.display = name === tab ? '' : 'none';
  });
  // The detail view belongs to Build, so Build stays lit in the nav.
  const navTab = tab === 'bracket' ? 'build' : tab;
  if (tab === 'explore') initExplore();
  document.querySelectorAll('.page-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === navTab);
  });
}

// ──────────────────────────────────────────────────────────────────
// BRACKET ENGINE
//
// All strategies share the same internal game format:
//   { round, region, team1, team2, win_prob, is_upset,
//     precomputed_winner_id? }   ← only present for pre-computed strategies
//
// Team objects:
//   { id, name, seed }
// ──────────────────────────────────────────────────────────────────

/* Build a display team object.
 *
 * Deliberately carries NO ratings. It used to attach barthag / adj_oe / adj_de,
 * which was what made a client-side log5 possible in the first place; every game
 * probability now arrives precomputed in the artifact, so shipping ratings to the
 * renderer would only be a loaded gun.
 */
/* Build a display team object.
 *
 * `rawName` is the artifact's canonical `teams[].name` (schema 5). Earlier
 * versions passed the id and title-cased it here, which cannot round-trip:
 * `saint_mary_s__ca` became "Saint Mary S Ca" and `tcu` became "Tcu". Names are
 * data, so they ship in the artifact — the same rule that moved game
 * probabilities out of the browser.
 *
 * Carries no ratings, deliberately: shipping them to the renderer is what made
 * a client-side log5 possible in the first place.
 */
function mkTeam(id, rawName, seed) {
  return { id, name: rawName.replace(/^\(\d+\)\s*/, ''), seed };
}



/* Every game already carries its winner.
 *
 * Brackets reach this renderer fully decided — chosen in Python, transported in
 * the artifact, expanded by candidateToRounds. The browser resolves nothing; the
 * client-side simulate()/pick() paths that used to live here were a second
 * implementation of the model and are gone.
 */
function gameWinner(game) {
  return game.precomputed_winner_id === game.team1.id ? game.team1 : game.team2;
}

// The only bracket the detail view ever shows is the one Build handed over.
function getRounds() {
  return window.GENERATED_ROUNDS || [];
}

// Build the champion's path through every round.
function championPath(rounds) {
  if (!rounds || rounds.length === 0) return { champ: null, path: [] };
  const champ = gameWinner(rounds[rounds.length - 1].games[0]);

  const path = [];
  for (const round of rounds) {
    for (const game of round.games) {
      const winner = gameWinner(game);
      if (winner.id !== champ.id) continue;
      const opp = winner.id === game.team1.id ? game.team2 : game.team1;
      const champWinProb = winner.id === game.team1.id ? game.win_prob : 1 - game.win_prob;
      path.push({
        round: round.round_name,
        opp_name: opp.name,
        opp_seed: opp.seed,
        win_prob: champWinProb,
      });
    }
  }
  return { champ, path };
}

// ── Actual-outcome grading (the 2026 tournament already happened — see
// actualData / docs/data/actual_2026.json) ──

// team_ids that actually won their game at this round, or null if we have
// no real-outcome data (e.g. actual_2026.json failed to load).
function actualWinnerSet(roundName) {
  if (!actualData) return null;
  const key = ROUND_TO_ACTUAL_KEY[roundName];
  return key ? new Set(actualData.results_by_round[key] || []) : null;
}

// Did `winner` (a team object) actually win their real round-`roundName`
// game? Returns null (unknown) rather than false when we have no data —
// callers must treat null as "don't render a verdict", not "wrong".
function isPickCorrect(winner, roundName) {
  const winners = actualWinnerSet(roundName);
  return winners ? winners.has(winner.id) : null;
}

// scoreAgainstActual() was removed: it computed the bracket's realized 2026
// score, which is a performance metric on an in-sample season. Deleted rather
// than left unreferenced, so it cannot be re-wired by a one-line change.

// ──────────────────────────────────────────────────────────────────
// RENDERING
// ──────────────────────────────────────────────────────────────────

/* Render the bracket Build handed over.
 *
 * Named for what it does now. The old activateStrategy() switched between
 * construction algorithms ("Pool Optimizer", "Region Beam Search"); there is one
 * bracket source now, so there is nothing to switch. */
function showGeneratedBracket() {
  currentRound = 'Round of 64';

  const rounds = getRounds();
  renderChampionPath(rounds);
  renderActualPanel(rounds);
  renderRoundTabs(rounds);
  renderGames(rounds);
}


// ── Strategy strip ──


// ── Champion path ──

function renderChampionPath(rounds) {
  const el = document.getElementById('path-flow');
  const { champ, path } = championPath(rounds);
  if (!champ) { el.innerHTML = '<p>No bracket data available.</p>'; return; }

  const champHTML = `
    <div class="path-champ">
      <div class="path-champ-seed">${champ.seed}</div>
      <div class="path-champ-name">${champ.name}</div>
      <div class="path-champ-label">Projected Champion</div>
    </div>`;

  const stepsHTML = path.map(stop => {
    const pct = (stop.win_prob * 100).toFixed(0);
    return `
      <div class="path-connector">→</div>
      <div class="path-stop">
        <div class="path-stop-round">${ROUND_SHORT[stop.round] || stop.round}</div>
        <div class="path-stop-opp">
          <span class="path-stop-seed">${stop.opp_seed}</span>${stop.opp_name}
        </div>
        <div class="path-stop-prob">${pct}%</div>
      </div>`;
  }).join('');

  el.innerHTML = champHTML + stepsHTML + '<div class="path-connector">→</div><div class="path-trophy">🏆</div>';
}

// ── Actual outcome panel ──
// The 2026 tournament this page displays picks for finished months ago.
// This renders how the displayed bracket actually
// scored against the real 2026 results — a retrospective, not a live
// pick recommendation. Hides itself entirely if actual_2026.json didn't
// load (degrades to the plain projected-picks view).

function renderActualPanel(rounds) {
  const section = document.getElementById('actual-section');
  const el = document.getElementById('actual-panel');
  if (!actualData) { section.style.display = 'none'; return; }
  section.style.display = '';

  /* NO SCORE IS SHOWN HERE.
   *
   * This panel used to grade the bracket against 2026 and display
   * "<points>/1920 pts", "<n>/63 picks correct" and a champion hit/miss badge.
   * Every one of those is a realized 2026 performance metric, and 2026 is an
   * in-sample integration fixture — spec 2027.v2 trains through it. Presenting
   * the model's score on a season it was built with is exactly the claim the
   * whole prospective discipline exists to prevent, and the production manifest
   * for this artifact states plainly that no 2026 outcome metric was computed.
   *
   * What remains is the factual outcome: who actually won. That is a fact about
   * the tournament, not a measure of the model. Per-game ✓/✗ marks on the board
   * are likewise factual — they say what happened, not how well the model did —
   * and no total is derived from them.
   */
  el.innerHTML = `
    <div class="actual-summary">
      <div class="actual-real">
        <div class="actual-real-label">Real 2026 champion</div>
        <div class="actual-real-value">🏆 ${actualData.champion_name}
          <span class="actual-real-sub">def. ${actualData.runner_up_name}</span>
        </div>
        <div class="actual-real-f4">Final Four: ${actualData.final_four_names.join(', ')}</div>
      </div>
    </div>
    <p class="actual-note">
      The 2026 tournament ended months before this page was built, so you can see
      what actually happened alongside the picks. No score is shown: the model was
      built with this season already in hand, so how it did here says nothing
      about how it would do on a season it has not seen.
    </p>`;
}

// ── Round tabs ──

function renderRoundTabs(rounds) {
  const el = document.getElementById('round-tabs');
  el.innerHTML = rounds.map(r => `
    <button class="round-btn${r.round_name === currentRound ? ' active' : ''}"
            onclick="selectRound('${r.round_name}')">
      ${ROUND_SHORT[r.round_name] || r.round_name}
    </button>`).join('');
}

function selectRound(round) {
  currentRound = round;
  const rounds = getRounds();
  renderRoundTabs(rounds);
  renderGames(rounds);
}

// ── Games grid ──

function renderGames(rounds) {
  const el     = document.getElementById('bracket-body');
  const round  = rounds.find(r => r.round_name === currentRound);
  if (!round) { el.innerHTML = ''; return; }

  const isNational = currentRound === 'Final Four' || currentRound === 'Championship';

  if (isNational) {
    el.innerHTML = `<div class="games-grid">${round.games.map(g => gameCard(g)).join('')}</div>`;
  } else {
    const grouped = {};
    for (const g of round.games) {
      if (!grouped[g.region]) grouped[g.region] = [];
      grouped[g.region].push(g);
    }
    el.innerHTML = REGIONS.filter(r => grouped[r]).map(region => `
      <div class="region-group">
        <div class="region-label">${region}</div>
        <div class="games-grid">${grouped[region].map(g => gameCard(g)).join('')}</div>
      </div>`).join('');
  }
}

// ── Game card ──

function gameCard(game) {
  const winner     = gameWinner(game);
  const t1IsPick   = winner.id === game.team1.id;
  const t1ProbPct  = (game.win_prob * 100).toFixed(1);
  const t2ProbPct  = (100 - parseFloat(t1ProbPct)).toFixed(1);
  const pickBadge  = '<span class="pick-badge">PICK</span>';
  // Upset: the bracket takes the higher-seeded (underdog) team
  const isUpsetPick = game.team1.seed !== game.team2.seed &&
    winner.id === (game.team1.seed > game.team2.seed ? game.team1.id : game.team2.id);

  const correct = isPickCorrect(winner, game.round);
  const verdictBadge = correct == null
    ? ''
    : correct
      ? '<span class="verdict-badge verdict-hit" title="This pick actually happened">✓ correct</span>'
      : '<span class="verdict-badge verdict-miss" title="This is not what actually happened">✗ missed</span>';

  return `
    <div class="game-card${isUpsetPick ? ' upset' : ''}${correct === false ? ' verdict-wrong' : ''}">
      <div class="game-team${t1IsPick ? ' is-pick' : ' not-pick'}">
        <div class="game-team-main">
          <span class="team-seed ${seedCls(game.team1.seed)}">${game.team1.seed}</span>
          <span class="team-name">${game.team1.name}</span>
        </div>
        <div class="game-team-right">
          ${t1IsPick ? pickBadge : ''}
          <div class="prob-stack">
            <span class="win-prob ${probCls(parseFloat(t1ProbPct))}">${t1ProbPct}%</span>
          </div>
        </div>
      </div>
      <div class="game-team${t1IsPick ? ' not-pick' : ' is-pick'}">
        <div class="game-team-main">
          <span class="team-seed ${seedCls(game.team2.seed)}">${game.team2.seed}</span>
          <span class="team-name">${game.team2.name}</span>
        </div>
        <div class="game-team-right">
          ${!t1IsPick ? pickBadge : ''}
          <div class="prob-stack">
            <span class="win-prob ${probCls(parseFloat(t2ProbPct))}">${t2ProbPct}%</span>
          </div>
        </div>
      </div>
      <div class="game-meta">
        <span>${game.region}</span>
        ${isUpsetPick ? '<span class="upset-badge">Upset pick</span>' : ''}
        ${verdictBadge}
      </div>
    </div>`;
}

// Small "what the opponent field actually picked" subtext under win_prob.
// null/undefined means no opponent data for this team/round — render nothing
// rather than a misleading 0%.


// ── Helpers ──

function seedCls(seed) {
  if (seed <= 4) return 'top-seed';
  if (seed <= 8) return 'mid-seed';
  return 'low-seed';
}

function probCls(pct) {
  if (pct >= 70) return 'high';
  if (pct >= 52) return 'mid';
  return 'low';
}


function pct(v) { return `${(v * 100).toFixed(1)}%`; }









// ──────────────────────────────────────────────────────────────────
// MATCHUP TABLE
//
// (scripts/generate_matchup_table.py). Same interaction model as the team
// stats table above — year select, search, click-to-sort — but the rows are
// GAMES rather than teams, because "my offense vs your defense" is a
// property of a pair and has no meaning as a per-team column.
// ──────────────────────────────────────────────────────────────────









// ──────────────────────────────────────────────────────────────────
// MODEL ACCURACY TAB
//
// Renders docs/data/ml_backtest.json — leave-one-year-out prediction
// metrics. Everything displayed is precomputed by
// scripts/generate_ml_backtest_data.py from the per-game prediction
// artifact; this file only formats it. The framing is deliberately
// unflattering where the data is unflattering (accuracy ties the seed
// baseline, several model families score negative skill, the market edges
// the model out) — see the caveats block in index.html.
// ──────────────────────────────────────────────────────────────────

// btPct tolerates null (unlike the bracket tab's pct(), which assumes a value).
const btPct = (v, d = 1) => v == null ? '—' : `${(v * 100).toFixed(d)}%`;
const num4  = (v) => v == null ? '—' : v.toFixed(4);

