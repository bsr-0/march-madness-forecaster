/* ══════════════════════════════════════════════════════════════════
 * March Madness Forecaster — Bracket Picker
 * ══════════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────────
// STRATEGY CATALOG
//
// This is the single source of truth for what appears in the
// strategy selector. Keep it accurate — it directly drives the UI.
//
// ═══ CURRENT BEST MODELS (validated 2026-08-16, 15-yr LOYO) ═════
//   Best:   meta_region_poolaware — 11.3% P(1st), 15-yr LOYO, 15/15 vs seed
//   Strong: meta_region           —  6.3% P(1st), 15-yr LOYO, 8/15 vs seed
//   Strong: meta_exhaustive        —  6.2% P(1st), 15-yr LOYO, 8/15 vs seed
//   Seed baseline (random-ish):      4.9% P(1st)
//   Canonical contract + full numbers: MEMORY.md §3 "Pool backtest —
//   production strategy validation". Supersedes the old 14-yr/11.9%/
//   3.1%-seed figures — those predate 2026 being added as a backtest
//   year (BACKTEST_YEARS in scripts/mc_pool_backtest.py now spans
//   15 years, 2011-2026 excl. 2020, now that 2026 has concluded).
//
// NOTE (2026-08-15): the 2026 tournament this page displays picks for
// finished months ago (real champion: Michigan, beat UConn 69-63 —
// see data/raw/historical/tournament_results_2026.json). This page is
// now a replay/demo, not a live pick tool.
//
// ═══ HOW TO UPDATE AFTER A NEW SEASON ══════════════════════════
//   1. Approve and run:  python scripts/mc_pool_backtest.py
//   2. Identify the best strategy from the printed backtest report
//   3. Update YEAR in scripts/generate_poolaware_bracket.py,
//      scripts/generate_region_bracket.py, and
//      scripts/generate_exhaustive_bracket.py, then run each —
//      they write docs/data/bracket_<YEAR>.json,
//      bracket_<YEAR>_region.json, and bracket_<YEAR>_exhaustive.json.
//      (Wired into .github/workflows/generate-web-data.yml so this
//      also happens automatically on data refreshes.)
//   4. Update the 'pool' entry below:
//      - Change bracket_file to point at the new JSON
//      - Update p_first and backtest_note with the new numbers
//   5. If a new strategy overtakes 'pool' as best, mark the old
//      one is_top: false and the new one is_top: true
//   6. Never hardcode a P(1st) number without a backtest citation
//
// ═══ ADDING A NEW PREFERENCE LENS ══════════════════════════════
//   Add an entry with a pick function (t1, t2, winProb) → team.
//   Set is_model: false and p_first: null if untested.
//   Mark badge_tone: 'sky' for stat lenses, 'red' for risky modes.
// ──────────────────────────────────────────────────────────────────

const STRATEGIES = [
  {
    key: 'pool',
    label: 'Pool Optimizer',
    subtitle: 'Best backtested model',
    description:
      'Simulates ~25 candidate brackets against a realistic pool field and picks the best one. ' +
      'Strongest strategy in 15 years of backtests.',
    p_first: 11.3,          // ← update this after each backtest run
    badge: '11.3% P(1st)',
    badge_tone: 'gold',
    is_top: true,           // ← set to true for the current best model
    is_model: true,
    backtest_note: '15/15 years beating seed, 15-yr backtest, N=31 pool.',
    // Pre-computed picks: pick === null means use winner_id from bracket_2026.json.
    // To swap to a new season's bracket, update the JSON file and re-deploy.
    pick: null,
  },
  {
    key: 'exhaustive',
    label: 'Exhaustive Search',
    subtitle: 'Best champion by simulation',
    description: 'Tests all 64 possible champions and picks whichever builds the highest-scoring bracket.',
    p_first: 6.2,
    badge: '~6.2% P(1st)',
    badge_tone: 'green',
    is_top: false,
    is_model: true,
    backtest_note: '8/15 years beating seed (gate boundary), 15-yr backtest, N=31 pool. Champion: Michigan.',
    pick: null,
  },
  {
    key: 'stat',
    label: 'Region Beam Search',
    subtitle: 'Region-level construction',
    description: 'Builds each region independently via beam search, then assembles the champion from the winners.',
    p_first: 6.3,
    badge: '~6.3% P(1st)',
    badge_tone: 'green',
    is_top: false,
    is_model: true,
    backtest_note: '8/15 years beating seed (gate boundary), 15-yr backtest, N=31 pool.',
    // Pre-computed picks: pick === null means use winner_id from bracket_2026_region.json.
    pick: null,
  },
  {
    key: 'chalk',
    label: 'Chalk',
    subtitle: 'Always the favorite',
    description: 'Picks the lower seed every game. Simple and safe, but no edge in a winner-take-all pool.',
    p_first: null,
    badge: 'Traditional',
    badge_tone: 'neutral',
    is_top: false,
    is_model: false,
    backtest_note: 'Matches seed baseline (~4.9% P(1st)). No edge in winner-take-all pools.',
    pick: (t1, t2) => {
      if (t1.seed !== t2.seed) return t1.seed < t2.seed ? t1 : t2;
      return t1.barthag >= t2.barthag ? t1 : t2;   // tie-break
    },
  },
];

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

// Colors per strategy badge tone (inline styles avoid needing a CSS class per strategy)
const BADGE_COLORS = {
  gold:    { bg: 'rgba(200,145,37,0.18)',  text: '#6b4800' },
  green:   { bg: 'rgba(47,122,99,0.16)',   text: '#1d5940' },
  red:     { bg: 'rgba(187,77,45,0.16)',   text: '#6e2211' },
  sky:     { bg: 'rgba(47,94,138,0.14)',   text: '#20405e' },
  neutral: { bg: 'rgba(92,64,41,0.10)',    text: '#4f3720' },
};

// ── App state ──

// bracketData[approach] — one precomputed Torvik-based bracket JSON per
// approach (pool, exhaustive, stat).
let bracketData = { pool: null, exhaustive: null, stat: null };
let loyoData          = null;   // per-year ESPN points, see docs/data/loyo_points.json
let actualData         = null;  // real 2026 outcome, see docs/data/actual_2026.json — the
                                 // 2026 tournament already concluded, this is a replay
let window3yrData      = null;  // 2024-2026 recency-fit backtest window, see docs/data/loyo_window_3yr_recency_fit.json
let currentWindow      = '15yr';   // '15yr' | '3yr' — which backtest window's P(1st)/note to display
let teamIndex        = {};      // team_id → { barthag, adj_oe, adj_de, champ_prob, elo_rating }
let currentKey       = 'pool';
let currentRound  = 'Round of 64';
let roundsCache   = {};         // key → computed rounds[]

// Multi-year pre-tournament team stats table, see docs/data/team_stats_by_year.json
let teamStatsData    = null;    // { years, generated, stats_by_year: { "2026": [row, ...], ... } }
let statsCurrentYear = null;    // set to the most recent year once teamStatsData loads
let statsSortColumn  = 't_rank';
let statsSortDir     = 'asc';   // 'asc' | 'desc'
let statsSearchQuery = '';

// ──────────────────────────────────────────────────────────────────
// BOOT
// ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  let profiles;
  try {
    const [poolTv, exhaustiveTv, regionTv, profilesRes] = await Promise.all([
      fetch('data/bracket_2026.json?v=2026-08-16d').then(r => r.json()),
      fetch('data/bracket_2026_exhaustive.json?v=2026-08-16d').then(r => r.json()),
      fetch('data/bracket_2026_region.json?v=2026-08-16d').then(r => r.json()),
      fetch('data/team_profiles.json?v=2026-08-16d').then(r => r.json()),
    ]);
    bracketData.pool       = poolTv;
    bracketData.exhaustive = exhaustiveTv;
    bracketData.stat       = regionTv;
    profiles = profilesRes;
  } catch (err) {
    document.body.innerHTML =
      '<p style="padding:48px;font-family:sans-serif;color:#bb4d2d">Failed to load bracket data. ' +
      'Make sure bracket_2026.json, bracket_2026_exhaustive.json, bracket_2026_region.json, ' +
      'and team_profiles.json are present in docs/data/.</p>';
    return;
  }

  // The per-year points panel and real-outcome retrospective are nice-to-
  // haves, not required to render the bracket picker — fetch separately so
  // a missing/broken file degrades to "no panel" instead of blocking the
  // whole page.
  let loyo;
  try {
    loyo = await fetch('data/loyo_points.json?v=2026-08-16d').then(r => r.json());
  } catch (err) {
    loyo = null;
  }
  let actual;
  try {
    actual = await fetch('data/actual_2026.json?v=2026-08-16d').then(r => r.json());
  } catch (err) {
    actual = null;
  }
  let window3yr;
  try {
    window3yr = await fetch('data/loyo_window_3yr_recency_fit.json?v=2026-08-17a').then(r => r.json());
  } catch (err) {
    window3yr = null;
  }
  let teamStats;
  try {
    teamStats = await fetch('data/team_stats_by_year.json?v=2026-08-17c').then(r => r.json());
  } catch (err) {
    teamStats = null;
  }

  loyoData       = loyo;
  actualData     = actual;
  window3yrData  = window3yr;
  teamStatsData  = teamStats;

  // Build O(1) team lookup
  for (const t of profiles.teams) {
    teamIndex[t.team_id] = {
      barthag:    t.barthag,
      adj_oe:     t.adj_offensive_efficiency,
      adj_de:     t.adj_defensive_efficiency,
      champ_prob: t.championship_prob ?? null,
      elo_rating: t.rating ?? null,
    };
  }

  renderStrategyStrip();
  renderWindowToggle();
  activateStrategy('pool');

  if (teamStatsData && teamStatsData.years && teamStatsData.years.length) {
    statsCurrentYear = teamStatsData.years[teamStatsData.years.length - 1];
    document.getElementById('stats-table-section').style.display = '';
    renderStatsYearSelect();
    renderStatsTable();
  }
});

// ──────────────────────────────────────────────────────────────────
// PAGE TABS
//
// Top-level tabs: "Bracket Picker" (the existing strategy/bracket UI) and
// "Team Stats" (the multi-year stats table). Client-side show/hide only —
// both tabs' DOM is always built at boot, this just toggles which one is
// visible, so switching tabs is instant with no re-fetch.
// ──────────────────────────────────────────────────────────────────

function setActiveTab(tab) {
  document.getElementById('tab-bracket').style.display = tab === 'bracket' ? '' : 'none';
  document.getElementById('tab-stats').style.display = tab === 'stats' ? '' : 'none';
  document.querySelectorAll('.page-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
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
//   { id, name, seed, barthag, adj_oe, adj_de, champ_prob, elo_rating }
// ──────────────────────────────────────────────────────────────────

// Log5 matchup probability given two Barthag values (p = P(beat avg D1)).
function log5(p1, p2) {
  const n = p1 * (1 - p2);
  const d = n + p2 * (1 - p1);
  return d > 0 ? n / d : 0.5;
}

// Build a rich team object, falling back to inline rating if not in teamIndex.
function mkTeam(id, rawName, seed, ratingFallback) {
  const prof = teamIndex[id] || {};
  return {
    id,
    name:           rawName.replace(/^\(\d+\)\s*/, ''),
    seed,
    barthag:        prof.barthag        ?? ratingFallback,
    adj_oe:         prof.adj_oe         ?? null,
    adj_de:         prof.adj_de         ?? null,
    champ_prob:     prof.champ_prob     ?? null,
    elo_rating:     prof.elo_rating     ?? null,
  };
}

// Convert a pre-computed bracket JSON into the internal game format. Picks
// AND win_prob both come straight from the JSON — it was actually
// constructed under Torvik round_probs, not recomputed client-side.
function precomputedRounds(data) {
  return data.rounds.map(round => ({
    round_name: round.round_name,
    games: round.games.map(g => ({
      round:   g.round,
      region:  g.region,
      team1:   mkTeam(g.team1_id, g.team1, g.team1_seed, g.team1_rating),
      team2:   mkTeam(g.team2_id, g.team2, g.team2_seed, g.team2_rating),
      win_prob: g.win_prob,
      is_upset: g.is_upset,
      precomputed_winner_id: g.winner_id,
      team1_pool_pct: g.team1_pool_pct ?? null,
      team2_pool_pct: g.team2_pool_pct ?? null,
    })),
  }));
}

function poolRounds()       { return precomputedRounds(bracketData.pool); }
function exhaustiveRounds() { return precomputedRounds(bracketData.exhaustive); }
function regionRounds()     { return precomputedRounds(bracketData.stat); }

// Precomputed strategies read their bracket straight from a JSON file
// instead of simulating client-side (see STRATEGIES pick === null).
const PRECOMPUTED_ROUNDS = {
  pool: poolRounds,
  exhaustive: exhaustiveRounds,
  stat: regionRounds,
};

// Simulate the full bracket for Chalk (the only strategy without a
// precomputed bracket) from R64. R64 matchup structure and win_prob come
// straight from the Torvik-built pool bracket.
function simulate(strategy) {
  const r64 = bracketData.pool.rounds[0].games.map(g => {
    const team1 = mkTeam(g.team1_id, g.team1, g.team1_seed, g.team1_rating);
    const team2 = mkTeam(g.team2_id, g.team2, g.team2_seed, g.team2_rating);
    return {
      round:    'Round of 64',
      region:   g.region,
      team1,
      team2,
      win_prob: g.win_prob,
      is_upset: g.is_upset,
    };
  });

  const allRounds = [{ round_name: 'Round of 64', games: r64 }];

  // Group R64 games by region for bracket progression
  let byRegion = {};
  for (const reg of REGIONS) byRegion[reg] = r64.filter(g => g.region === reg);

  // R32, S16, E8 — each round halves games per region
  for (const roundName of ['Round of 32', 'Sweet 16', 'Elite 8']) {
    const nextGames = [];
    const nextByRegion = {};

    for (const reg of REGIONS) {
      const prev = byRegion[reg];
      nextByRegion[reg] = [];

      for (let i = 0; i < prev.length; i += 2) {
        const w1 = pick(prev[i], strategy);
        const w2 = pick(prev[i + 1], strategy);
        const wp = log5(w1.barthag, w2.barthag);
        const game = {
          round: roundName, region: reg,
          team1: w1, team2: w2,
          win_prob: wp,
          is_upset: upsetCheck(w1, w2, wp),
        };
        nextByRegion[reg].push(game);
        nextGames.push(game);
      }
    }

    allRounds.push({ round_name: roundName, games: nextGames });
    byRegion = nextByRegion;
  }

  // Final Four: East vs West, South vs Midwest
  const e8w = {};
  for (const reg of REGIONS) e8w[reg] = pick(byRegion[reg][0], strategy);

  const f4 = [
    mkGame('Final Four', 'East vs West',    e8w['East'],  e8w['West'],    strategy),
    mkGame('Final Four', 'South vs Midwest', e8w['South'], e8w['Midwest'], strategy),
  ];
  allRounds.push({ round_name: 'Final Four', games: f4 });

  const champ = mkGame('Championship', 'Championship',
    pick(f4[0], strategy), pick(f4[1], strategy), strategy);
  allRounds.push({ round_name: 'Championship', games: [champ] });

  return allRounds;
}

function mkGame(round, region, t1, t2, strategy) {
  const wp = log5(t1.barthag, t2.barthag);
  return { round, region, team1: t1, team2: t2, win_prob: wp, is_upset: upsetCheck(t1, t2, wp) };
}

function upsetCheck(t1, t2, wp) {
  return (t1.seed > t2.seed && wp > 0.5) || (t2.seed > t1.seed && wp < 0.5);
}

// Determine the winner of a game for a given strategy.
function pick(game, strategy) {
  // Pool strategy: use pre-computed result
  if (game.precomputed_winner_id != null) {
    return game.precomputed_winner_id === game.team1.id ? game.team1 : game.team2;
  }
  if (strategy.pick) {
    return strategy.pick(game.team1, game.team2, game.win_prob);
  }
  return game.team1.barthag >= game.team2.barthag ? game.team1 : game.team2;
}

// Get cached rounds for a strategy key.
function getRounds(key) {
  if (roundsCache[key]) return roundsCache[key];
  const s = STRATEGIES.find(s => s.key === key);
  if (s.pick === null) {
    roundsCache[key] = PRECOMPUTED_ROUNDS[key]();
  } else {
    roundsCache[key] = simulate(s);
  }
  return roundsCache[key];
}

// Build the champion's path through every round.
function championPath(rounds, key) {
  if (!rounds || rounds.length === 0) return { champ: null, path: [] };
  const strategy = STRATEGIES.find(s => s.key === key);
  const champGame = rounds[rounds.length - 1].games[0];
  const champ = pick(champGame, strategy);

  const path = [];
  for (const round of rounds) {
    for (const game of round.games) {
      const winner = pick(game, strategy);
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

// Score every pick in `rounds` against the real outcome. Returns null if
// actual_2026.json isn't loaded. `points`/`maxPoints` use the same
// ESPN_SCORING weights the backtest is validated against.
function scoreAgainstActual(rounds, strategy) {
  if (!actualData) return null;
  let points = 0, maxPoints = 0, correct = 0, total = 0;
  for (const round of rounds) {
    const key = ROUND_TO_ACTUAL_KEY[round.round_name];
    const winners = key ? actualWinnerSet(round.round_name) : null;
    const perPick = key ? (actualData.scoring[key] || 0) : 0;
    for (const game of round.games) {
      if (!winners) continue;
      total++;
      maxPoints += perPick;
      const winner = pick(game, strategy);
      if (winners.has(winner.id)) { correct++; points += perPick; }
    }
  }
  return { points, maxPoints, correct, total };
}

// ──────────────────────────────────────────────────────────────────
// RENDERING
// ──────────────────────────────────────────────────────────────────

function activateStrategy(key) {
  currentKey   = key;
  currentRound = 'Round of 64';

  renderStrategyStrip();
  renderStrategyDetail();

  const rounds = getRounds(key);
  renderChampionPath(rounds, key);
  renderActualPanel(rounds, key);
  renderRoundTabs(rounds);
  renderGames(rounds);
}

// ── Backtest window toggle ──
//
// STRATEGIES bakes in the 15-year (2011-2026) LOYO numbers as the default.
// window3yrData (docs/data/loyo_window_3yr_recency_fit.json) holds a
// second, much lower-power (n=3) cut over just 2024-2026. Unlike an earlier
// version of this toggle, the '3yr' window is a genuine re-optimization,
// not just a re-windowed display of the same 15-yr picks: the pool
// strategy's blend_alpha hyperparameter is re-fit using only the most
// recent 3 walk-forward years (src/optimization/recency_hparam_fitter.py),
// scored via the same MC-pool-simulation P(1st) estimator production
// selection uses. meta_region/meta_exhaustive never use the "blend"
// probability base at all, so they're structurally unaffected and their
// entries here are identical to what a 15-yr-style run over just these 3
// years would show — only 'pool' actually differs.
//
// Diagnostic only either way — it does NOT change what bracket actually
// gets submitted (scripts/generate_poolaware_bracket.py, the live 2026
// production script, is untouched).
const WINDOW_DEFS = {
  '15yr': { label: '15-Year (2011–2026)', note: 'Full backtest, N=31 pool. Statistically validated.' },
  '3yr':  { label: '2024–2026 (3-yr, refit)', note: 'Diagnostic only — pool strategy re-optimized on the 3-yr window itself, too few years for significance testing.' },
};

function setWindow(key) {
  currentWindow = key;
  renderWindowToggle();
  renderStrategyStrip();
  renderStrategyDetail();
}

function renderWindowToggle() {
  const toggleEl = document.getElementById('window-toggle');
  const noteEl   = document.getElementById('window-note');
  if (!toggleEl) return;

  toggleEl.innerHTML = Object.entries(WINDOW_DEFS).map(([key, def]) => `
    <button class="window-toggle-btn${key === currentWindow ? ' active' : ''}"
            onclick="setWindow('${key}')">${def.label}</button>
  `).join('');

  if (noteEl) noteEl.textContent = WINDOW_DEFS[currentWindow].note;
}

// Resolve the P(1st)/badge/note to display for a strategy under the
// currently selected backtest window. Falls back to the strategy's baked-in
// 15-yr numbers if a given window has no entry for it (e.g. Chalk, which
// isn't backtested as a standalone strategy in any window).
function effectiveStrategyStats(key) {
  const s = STRATEGIES.find(s => s.key === key);
  if (currentWindow === '3yr' && window3yrData && window3yrData.strategies[key]) {
    const w = window3yrData.strategies[key];
    return { p_first: w.p_first, badge: `~${w.p_first}% P(1st)`, backtest_note: w.note };
  }
  return { p_first: s.p_first, badge: s.badge, backtest_note: s.backtest_note };
}

// ── Strategy strip ──

function renderStrategyStrip() {
  const el = document.getElementById('strategy-strip');
  el.innerHTML = STRATEGIES.map(s => {
    const bc = BADGE_COLORS[s.badge_tone] || BADGE_COLORS.neutral;
    const stats = effectiveStrategyStats(s.key);
    const badgeHTML = stats.badge
      ? `<span class="strategy-badge" style="background:${bc.bg};color:${bc.text}">${stats.badge}</span>`
      : '';
    const starHTML = s.is_top ? '<span class="strategy-star">★</span>' : '';
    return `
      <button class="strategy-btn${s.key === currentKey ? ' active' : ''}"
              onclick="activateStrategy('${s.key}')">
        ${starHTML}
        <span class="strategy-btn-label">${s.label}</span>
        <span class="strategy-btn-sub">${s.subtitle}</span>
        ${badgeHTML}
      </button>`;
  }).join('');
}

// ── Strategy detail panel ──

// Source data behind each precomputed strategy's team1_pool_pct/team2_pool_pct
// annotations — "what fraction of the opponent field picked this team."
const OPPONENT_SOURCE_DATA = {
  pool:       () => bracketData.pool,
  exhaustive: () => bracketData.exhaustive,
  stat:       () => bracketData.stat,
};

function renderStrategyDetail() {
  const el = document.getElementById('strategy-detail');
  const s  = STRATEGIES.find(s => s.key === currentKey);
  const bc = BADGE_COLORS[s.badge_tone] || BADGE_COLORS.neutral;

  const tagHTML = s.is_model
    ? `<span class="sd-tag sd-tag-model">Model-backed</span>`
    : `<span class="sd-tag sd-tag-pref">Preference lens</span>`;

  const stats = effectiveStrategyStats(s.key);
  const perfHTML = stats.p_first != null
    ? `<span class="sd-perf" style="background:${bc.bg};color:${bc.text}">${stats.p_first}% P(1st) historically</span>`
    : '';

  const dataFn = OPPONENT_SOURCE_DATA[currentKey];
  const opponentSource = dataFn ? dataFn()?.opponent_source : null;
  const opponentHTML = opponentSource
    ? `<p class="sd-opponent">"% of pool" below is from <strong>${opponentSource}</strong>.</p>`
    : '';

  el.innerHTML = `
    <div class="sd-row">${tagHTML}${perfHTML}</div>
    <p class="sd-desc">${s.description}</p>
    <p class="sd-note">${stats.backtest_note}</p>
    ${opponentHTML}
    ${loyoPointsHTML(currentKey)}
  `;
}

// Per-year ESPN points, one holdout year at a time (leave-one-year-out
// walk-forward backtest — the actual points that strategy's bracket would
// have scored against that year's real outcome, not a simulated proxy).
function loyoPointsHTML(key) {
  if (!loyoData || !loyoData.points_by_strategy[key]) return '';
  const pts = { ...loyoData.points_by_strategy[key] };

  // Under the 3-yr (recency-fit) window, swap in the counterfactual
  // per-year scores the recency-fitted bracket actually would have scored
  // against each year's real outcome — not just a re-windowed view of the
  // same production picks. Only years where the fitter's chosen blend_alpha
  // differs from the production default (0.5) can actually differ; where it
  // doesn't, the override value is identical to the production one anyway.
  const overrideYears = new Set();
  if (currentWindow === '3yr' && window3yrData) {
    const perYear = (window3yrData.strategies[key] || {}).per_year_score;
    if (perYear) {
      for (const [y, v] of Object.entries(perYear)) {
        pts[y] = v;
        overrideYears.add(y);
      }
    }
  }

  const years = loyoData.years.filter(y => pts[y] != null);
  if (years.length === 0) return '';

  const vals = years.map(y => pts[y]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const max = Math.max(...vals);

  const chips = years.map(y => {
    const v = pts[y];
    const heightPct = Math.max(8, Math.round((v / max) * 100));
    const diagFlag = overrideYears.has(y) ? ' loyo-chip-diagnostic' : '';
    const title = overrideYears.has(y)
      ? `${y}: ${v.toFixed(0)} pts (3-yr recency-fit diagnostic)`
      : `${y}: ${v.toFixed(0)} pts`;
    return `
      <div class="loyo-chip${diagFlag}" title="${title}">
        <div class="loyo-chip-bar-track">
          <div class="loyo-chip-bar" style="height:${heightPct}%"></div>
        </div>
        <div class="loyo-chip-pts">${v.toFixed(0)}</div>
        <div class="loyo-chip-year">'${y.slice(2)}</div>
      </div>`;
  }).join('');

  return `
    <div class="loyo-points">
      <p class="loyo-points-label">Points by year — mean ${mean.toFixed(0)}</p>
      <div class="loyo-points-scroll">
        <div class="loyo-points-row">${chips}</div>
      </div>
    </div>`;
}

// ── Champion path ──

function renderChampionPath(rounds, key) {
  const el = document.getElementById('path-flow');
  const { champ, path } = championPath(rounds, key);
  if (!champ) { el.innerHTML = '<p>No bracket data available.</p>'; return; }
  const bc = BADGE_COLORS[STRATEGIES.find(s => s.key === key).badge_tone] || BADGE_COLORS.neutral;

  const champHTML = `
    <div class="path-champ">
      <div class="path-champ-seed" style="background:${bc.bg};color:${bc.text}">${champ.seed}</div>
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
// This renders how the currently-active strategy's bracket actually
// scored against the real 2026 results — a retrospective, not a live
// pick recommendation. Hides itself entirely if actual_2026.json didn't
// load (degrades to the plain projected-picks view).

function renderActualPanel(rounds, key) {
  const section = document.getElementById('actual-section');
  const el = document.getElementById('actual-panel');
  if (!actualData) { section.style.display = 'none'; return; }
  section.style.display = '';

  const strategy = STRATEGIES.find(s => s.key === key);
  const score = scoreAgainstActual(rounds, strategy);
  const { champ } = championPath(rounds, key);
  const champHit = champ && champ.id === actualData.champion_id;

  el.innerHTML = `
    <div class="actual-summary">
      <div class="actual-real">
        <div class="actual-real-label">Real 2026 champion</div>
        <div class="actual-real-value">🏆 ${actualData.champion_name}
          <span class="actual-real-sub">def. ${actualData.runner_up_name}</span>
        </div>
        <div class="actual-real-f4">Final Four: ${actualData.final_four_names.join(', ')}</div>
      </div>
      <div class="actual-score">
        <div class="actual-score-value">${score.points} / ${score.maxPoints} pts</div>
        <div class="actual-score-sub">${score.correct}/${score.total} picks correct</div>
        <div class="actual-score-champ ${champHit ? 'hit' : 'miss'}">
          ${champHit ? '✓ Picked the real champion' : `✗ Picked ${champ ? champ.name : '—'}, not ${actualData.champion_name}`}
        </div>
      </div>
    </div>
    <p class="actual-note">
      The 2026 tournament ended months before this page was built — this is a graded replay of
      <strong>${strategy.label}</strong>'s bracket against what actually happened, not a live prediction.
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
  const rounds = getRounds(currentKey);
  renderRoundTabs(rounds);
  renderGames(rounds);
}

// ── Games grid ──

function renderGames(rounds) {
  const el     = document.getElementById('bracket-body');
  const round  = rounds.find(r => r.round_name === currentRound);
  if (!round) { el.innerHTML = ''; return; }

  const strategy = STRATEGIES.find(s => s.key === currentKey);
  const isNational = currentRound === 'Final Four' || currentRound === 'Championship';

  if (isNational) {
    el.innerHTML = `<div class="games-grid">${round.games.map(g => gameCard(g, strategy)).join('')}</div>`;
  } else {
    const grouped = {};
    for (const g of round.games) {
      if (!grouped[g.region]) grouped[g.region] = [];
      grouped[g.region].push(g);
    }
    el.innerHTML = REGIONS.filter(r => grouped[r]).map(region => `
      <div class="region-group">
        <div class="region-label">${region}</div>
        <div class="games-grid">${grouped[region].map(g => gameCard(g, strategy)).join('')}</div>
      </div>`).join('');
  }
}

// ── Game card ──

function gameCard(game, strategy) {
  const winner     = pick(game, strategy);
  const t1IsPick   = winner.id === game.team1.id;
  const t1ProbPct  = (game.win_prob * 100).toFixed(1);
  const t2ProbPct  = (100 - parseFloat(t1ProbPct)).toFixed(1);
  const bc         = BADGE_COLORS[strategy.badge_tone] || BADGE_COLORS.neutral;
  const pickBadge  = `<span class="pick-badge" style="background:${bc.bg};color:${bc.text}">PICK</span>`;
  // Upset: strategy is picking the higher-seeded (underdog) team
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
            ${poolPctLabel(game.team1_pool_pct)}
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
            ${poolPctLabel(game.team2_pool_pct)}
          </div>
        </div>
      </div>
      ${miniStats(game.team1, game.team2)}
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
function poolPctLabel(pct) {
  if (pct == null) return '';
  return `<span class="pool-pct">${(pct * 100).toFixed(0)}% of pool</span>`;
}

function miniStats(t1, t2) {
  const defs = [
    { label: 'Barthag', v1: t1.barthag, v2: t2.barthag, higherWins: true,  fmt: v => v.toFixed(3) },
    { label: 'Adj OE',  v1: t1.adj_oe,  v2: t2.adj_oe,  higherWins: true,  fmt: v => v.toFixed(1) },
    { label: 'Adj DE',  v1: t1.adj_de,  v2: t2.adj_de,  higherWins: false, fmt: v => v.toFixed(1) },
  ].filter(d => d.v1 != null && d.v2 != null);

  if (defs.length === 0) return '';

  const items = defs.map(d => {
    const t1Better = d.higherWins ? d.v1 > d.v2 : d.v1 < d.v2;
    return `
      <div class="mini-stat">
        <span class="mini-stat-label">${d.label}</span>
        <span class="mini-stat-pair">
          <span class="${t1Better ? 'ms-edge' : ''}">${d.fmt(d.v1)}</span>
          <span class="ms-sep">·</span>
          <span class="${!t1Better ? 'ms-edge' : ''}">${d.fmt(d.v2)}</span>
        </span>
      </div>`;
  }).join('');

  return `<div class="game-mini-stats">${items}</div>`;
}

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

// ──────────────────────────────────────────────────────────────────
// TEAM STATS TABLE
//
// Multi-year pre-tournament Torvik stats browser, see
// docs/data/team_stats_by_year.json (scripts/generate_team_stats_table.py).
// Single source of truth for the table's columns — header, cell
// rendering, and sorting all derive from this list.
// ──────────────────────────────────────────────────────────────────

const STATS_COLUMNS = [
  { key: 'team_name',                 label: 'Team',        numeric: false },
  { key: 'seed',                      label: 'Seed',         numeric: true },
  { key: 'region',                    label: 'Region',       numeric: false },
  { key: 'conference',                label: 'Conf',          numeric: false },
  { key: 't_rank',                    label: 'Rank',          numeric: true },
  { key: 'barthag',                   label: 'Barthag',       numeric: true, fmt: v => v.toFixed(3) },
  { key: 'adj_offensive_efficiency',  label: 'Adj OE',        numeric: true, fmt: v => v.toFixed(1) },
  { key: 'adj_defensive_efficiency',  label: 'Adj DE',        numeric: true, fmt: v => v.toFixed(1) },
  { key: 'adj_tempo',                 label: 'Tempo',         numeric: true, fmt: v => v.toFixed(1) },
  { key: 'effective_fg_pct',          label: 'eFG%',          numeric: true, fmt: pct },
  { key: 'turnover_rate',             label: 'TO%',           numeric: true, fmt: pct },
  { key: 'offensive_reb_rate',        label: 'OReb%',         numeric: true, fmt: pct },
  { key: 'free_throw_rate',           label: 'FTRate',        numeric: true, fmt: pct },
  { key: 'opp_effective_fg_pct',      label: 'Opp eFG%',      numeric: true, fmt: pct },
  { key: 'opp_turnover_rate',         label: 'Opp TO%',       numeric: true, fmt: pct },
  { key: 'defensive_reb_rate',        label: 'DReb%',         numeric: true, fmt: pct },
  { key: 'opp_free_throw_rate',       label: 'Opp FTRate',    numeric: true, fmt: pct },
  // Regular-season volatility, computed from the pre-tournament game log.
  { key: 'games_played',              label: 'G',             numeric: true },
  { key: 'reg_season_margin_avg',     label: 'Margin',        numeric: true, fmt: signed1 },
  { key: 'reg_season_margin_std',     label: 'Margin SD',     numeric: true, fmt: v => v.toFixed(1) },
  { key: 'close_game_rate',           label: 'Close%',        numeric: true, fmt: pct },
  { key: 'close_game_win_rate',       label: 'Close W%',      numeric: true, fmt: pct },
  { key: 'losses_to_weaker_rate',     label: 'Bad Loss%',     numeric: true, fmt: pct },
  // Post-hoc tournament result — NOT knowable before the tournament. Flagged
  // `outcome: true` so the renderer can visually fence it off from every
  // column above it (see .stats-table .outcome in style.css).
  { key: 'outcome_finish',            label: 'Finish',        numeric: false, outcome: true },
  { key: 'outcome_rounds_won',        label: 'Wins',          numeric: true,  outcome: true },
  { key: 'outcome_vs_seed_delta',     label: 'vs Seed',       numeric: true,  outcome: true, fmt: signed2 },
];

function pct(v) { return `${(v * 100).toFixed(1)}%`; }
function signed1(v) { return `${v > 0 ? '+' : ''}${v.toFixed(1)}`; }
function signed2(v) { return `${v > 0 ? '+' : ''}${v.toFixed(2)}`; }

function renderStatsYearSelect() {
  const el = document.getElementById('stats-year-select');
  if (!el || !teamStatsData) return;
  el.innerHTML = teamStatsData.years.map(y =>
    `<option value="${y}"${y === statsCurrentYear ? ' selected' : ''}>${y}</option>`
  ).join('');
}

function setStatsYear(year) {
  statsCurrentYear = Number(year);
  renderStatsTable();
}

function sortStatsBy(column) {
  if (statsSortColumn === column) {
    statsSortDir = statsSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    statsSortColumn = column;
    statsSortDir = STATS_COLUMNS.find(c => c.key === column).numeric ? 'asc' : 'asc';
  }
  renderStatsTable();
}

function setStatsSearch(query) {
  statsSearchQuery = query.trim().toLowerCase();
  renderStatsTable();
}

function filterStatsRows(rows, query) {
  if (!query) return rows;
  return rows.filter(r =>
    (r.team_name || '').toLowerCase().includes(query) ||
    (r.conference || '').toLowerCase().includes(query)
  );
}

function sortStatsRows(rows, column, dir) {
  const sorted = [...rows].sort((a, b) => {
    const av = a[column], bv = b[column];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;   // nulls sort last regardless of direction
    if (bv == null) return -1;
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
  });
  if (dir === 'desc') sorted.reverse();
  return sorted;
}

function renderStatsTable() {
  if (!teamStatsData || statsCurrentYear == null) return;
  const headEl = document.getElementById('stats-table-head');
  const bodyEl = document.getElementById('stats-table-body');
  if (!headEl || !bodyEl) return;

  // The first `outcome` column gets a divider so the post-hoc block is
  // unmistakably fenced off from the pre-tournament columns.
  const firstOutcome = STATS_COLUMNS.findIndex(c => c.outcome);
  const cls = (col, i) =>
    `${col.numeric ? ' numeric' : ''}${col.outcome ? ' outcome' : ''}${i === firstOutcome ? ' outcome-start' : ''}`;

  headEl.innerHTML = `<tr>${STATS_COLUMNS.map((col, i) => {
    const active = col.key === statsSortColumn;
    const arrow = active ? (statsSortDir === 'asc' ? ' ▲' : ' ▼') : '';
    const title = col.outcome ? ' title="Tournament result — known only after the fact, not a pre-tournament stat"' : '';
    return `<th class="sortable${active ? ' active' : ''}${cls(col, i)}"${title}
                onclick="sortStatsBy('${col.key}')">${col.label}${arrow}</th>`;
  }).join('')}</tr>`;

  const yearRows = teamStatsData.stats_by_year[String(statsCurrentYear)] || [];
  const filtered = filterStatsRows(yearRows, statsSearchQuery);
  const sorted = sortStatsRows(filtered, statsSortColumn, statsSortDir);

  if (sorted.length === 0) {
    bodyEl.innerHTML = `<tr><td colspan="${STATS_COLUMNS.length}" class="stats-empty">No teams match “${statsSearchQuery}”.</td></tr>`;
    return;
  }

  bodyEl.innerHTML = sorted.map(row => `<tr>${STATS_COLUMNS.map((col, i) => {
    const v = row[col.key];
    const display = v == null ? '—' : (col.fmt ? col.fmt(v) : v);
    return `<td class="${cls(col, i).trim()}">${display}</td>`;
  }).join('')}</tr>`).join('');
}

