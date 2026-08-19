/* ══════════════════════════════════════════════════════════════════
 * March Madness Forecaster — Bracket Picker
 * ══════════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────────
// STRATEGY CATALOG
//
// This is the single source of truth for what appears in the
// strategy selector. Keep it accurate — it directly drives the UI.
//
// ═══ CURRENT BEST MODELS (re-validated 2026-08-18, 15-yr LOYO) ═════
//   Best:   meta_region_poolaware — 11.2% P(1st), 15-yr LOYO, 15/15 vs seed
//   Strong: meta_region           —  6.3% P(1st), 15-yr LOYO, 14/15 vs seed (MeanRank)
//   Strong: meta_exhaustive        —  6.2% P(1st), 15-yr LOYO, 14/15 vs seed (MeanRank)
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
    p_first: 11.2,          // ← update this after each backtest run
    badge: '11.2% P(1st)',
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
    backtest_note: '14/15 years beating seed on MeanRank, 15-yr backtest, N=31 pool. Champion: Michigan.',
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
    backtest_note: '14/15 years beating seed on MeanRank, 15-yr backtest, N=31 pool.',
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
let mlBacktestData   = null;    // LOYO prediction-accuracy metrics, see docs/data/ml_backtest.json
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

// Pairwise tournament matchups, see docs/data/matchups_by_year.json
let matchupData        = null;   // { years, generated, matchups_by_year: { "2026": [game, ...] } }
let matchupYear        = null;
let matchupSortColumn  = 'fav_seed';
let matchupSortDir     = 'asc';
let matchupSearchQuery = '';

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
  let teamStats;
  try {
    teamStats = await fetch('data/team_stats_by_year.json?v=2026-08-17e').then(r => r.json());
  } catch (err) {
    teamStats = null;
  }
  let matchups;
  try {
    matchups = await fetch('data/matchups_by_year.json?v=2026-08-17e').then(r => r.json());
  } catch (err) {
    matchups = null;
  }
  let mlBacktest;
  try {
    mlBacktest = await fetch('data/ml_backtest.json?v=2026-08-18a').then(r => r.json());
  } catch (err) {
    mlBacktest = null;
  }

  loyoData        = loyo;
  actualData      = actual;
  teamStatsData   = teamStats;
  matchupData     = matchups;
  mlBacktestData  = mlBacktest;

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
  activateStrategy('pool');

  if (teamStatsData && teamStatsData.years && teamStatsData.years.length) {
    statsCurrentYear = teamStatsData.years[teamStatsData.years.length - 1];
    document.getElementById('stats-table-section').style.display = '';
    renderStatsYearSelect();
    renderStatsTable();
  }

  if (matchupData && matchupData.years && matchupData.years.length) {
    matchupYear = matchupData.years[matchupData.years.length - 1];
    document.getElementById('matchup-table-section').style.display = '';
    renderMatchupYearSelect();
    renderMatchupTable();
  }

  renderMlBacktest();
});

// ──────────────────────────────────────────────────────────────────
// PAGE TABS
//
// Top-level tabs: "Bracket Picker" (the existing strategy/bracket UI) and
// "Team Stats" (the multi-year stats table). Client-side show/hide only —
// both tabs' DOM is always built at boot, this just toggles which one is
// visible, so switching tabs is instant with no re-fetch.
// ──────────────────────────────────────────────────────────────────

const PAGE_TABS = ['bracket', 'stats', 'matchups', 'backtest'];

function setActiveTab(tab) {
  PAGE_TABS.forEach(name => {
    const el = document.getElementById(`tab-${name}`);
    if (el) el.style.display = name === tab ? '' : 'none';
  });
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

// Strategy stats always come from the full 15-year (2011-2026) LOYO
// backtest baked into STRATEGIES. A 3-year (2024-2026) recency-refit window
// toggle used to sit here as a diagnostic; it was removed because the refit
// didn't improve the score and the n=3 window was too low-power to test for
// significance. The fitter that produced it
// (src/optimization/recency_hparam_fitter.py) is retained for offline
// analysis — it never fed the submitted bracket either way.
function effectiveStrategyStats(key) {
  const s = STRATEGIES.find(s => s.key === key);
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
  const pts = loyoData.points_by_strategy[key];

  const years = loyoData.years.filter(y => pts[y] != null);
  if (years.length === 0) return '';

  const vals = years.map(y => pts[y]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const max = Math.max(...vals);

  const chips = years.map(y => {
    const v = pts[y];
    const heightPct = Math.max(8, Math.round((v / max) * 100));
    const title = `${y}: ${v.toFixed(0)} pts`;
    return `
      <div class="loyo-chip" title="${title}">
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
  { key: 'overtime_rate',             label: 'OT%',           numeric: true, fmt: pct },
  // Roster composition. Class and prior-roster membership are settled long
  // before March, so these are pre-tournament (see build_roster_stats).
  { key: 'returning_minutes_pct',     label: 'Returning%',    numeric: true, fmt: pct },
  { key: 'freshman_minutes_pct',      label: 'Frosh%',        numeric: true, fmt: pct },
  // Shooting profile and defensive pressure from the Kaggle regular-season
  // box score (MRegularSeasonDetailedResults) — contains zero NCAA tournament
  // games, so pre-tournament by construction; no date filter needed.
  { key: 'three_pt_rate',             label: '3PT Rate',      numeric: true, fmt: pct },
  { key: 'three_pt_pct',              label: '3P%',           numeric: true, fmt: pct },
  { key: 'opp_three_pt_pct',          label: 'Opp 3P%',       numeric: true, fmt: pct },
  { key: 'ast_to_ratio',              label: 'Ast/TO',        numeric: true, fmt: v => v.toFixed(2) },
  { key: 'havoc_rate',                label: 'Havoc/G',       numeric: true, fmt: v => v.toFixed(1) },
  { key: 'true_road_win_pct',         label: 'Road/Neut W%',  numeric: true, fmt: pct },
  // Program tournament history to date. Backward-looking only (prior years,
  // never the current one), so this IS pre-tournament information.
  { key: 'hist_residual',             label: 'Hist Resid',    numeric: true, fmt: signed2 },
  { key: 'hist_appearances',          label: 'Prior App',     numeric: true },
  // Head coach's tournament track record BEFORE this season (Kaggle
  // MTeamCoaches + MNCAATourneyCompactResults). Strictly backward-looking —
  // this year's own result is never included.
  { key: 'coach_name',                label: 'Coach',         numeric: false, fmt: v => v.split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' ') },
  { key: 'coach_prior_tourney_games', label: 'Coach G',       numeric: true },
  { key: 'coach_prior_tourney_wins',  label: 'Coach W',       numeric: true },
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


// ──────────────────────────────────────────────────────────────────
// MATCHUP TABLE
//
// Pairwise tournament matchups, see docs/data/matchups_by_year.json
// (scripts/generate_matchup_table.py). Same interaction model as the team
// stats table above — year select, search, click-to-sort — but the rows are
// GAMES rather than teams, because "my offense vs your defense" is a
// property of a pair and has no meaning as a per-team column.
// ──────────────────────────────────────────────────────────────────

const MATCHUP_COLUMNS = [
  { key: 'round',                 label: 'Round',      numeric: false },
  { key: 'region',                label: 'Region',     numeric: false },
  { key: 'fav_seed',              label: 'Favourite',  numeric: false,
    get: r => `(${r.fav_seed}) ${r.fav}`, sortKey: 'fav_seed' },
  { key: 'dog_seed',              label: 'Underdog',   numeric: false,
    get: r => `(${r.dog_seed}) ${r.dog}`, sortKey: 'dog_seed' },
  { key: 'barthag_diff',          label: 'Barthag Δ',  numeric: true, fmt: v => v.toFixed(3) },
  { key: 'fav_off_vs_dog_def',    label: 'Fav O vs Dog D', numeric: true, fmt: signed1 },
  { key: 'dog_off_vs_fav_def',    label: 'Dog O vs Fav D', numeric: true, fmt: signed1 },
  { key: 'fav_efg_vs_dog_def',    label: 'Fav eFG edge',   numeric: true, fmt: signedPct },
  { key: 'dog_efg_vs_fav_def',    label: 'Dog eFG edge',   numeric: true, fmt: signedPct },
  { key: 'fav_to_vs_dog_press',   label: 'Fav TO vs Press', numeric: true, fmt: signedPct },
  { key: 'dog_to_vs_fav_press',   label: 'Dog TO vs Press', numeric: true, fmt: signedPct },
  { key: 'fav_oreb_vs_dog_dreb',  label: 'Fav OReb edge',  numeric: true, fmt: signedPct },
  { key: 'dog_oreb_vs_fav_dreb',  label: 'Dog OReb edge',  numeric: true, fmt: signedPct },
  { key: 'tempo_diff',            label: 'Tempo Δ',    numeric: true, fmt: signed1 },
  { key: 'fav_margin_sd',         label: 'Fav SD',     numeric: true, fmt: v => v.toFixed(1) },
  { key: 'dog_margin_sd',         label: 'Dog SD',     numeric: true, fmt: v => v.toFixed(1) },
  { key: 'fav_close_win_pct',     label: 'Fav Close W%', numeric: true, fmt: pct },
  { key: 'dog_close_win_pct',     label: 'Dog Close W%', numeric: true, fmt: pct },
  { key: 'result_winner',         label: 'Winner',     numeric: false, outcome: true },
  { key: 'result_score',          label: 'Score',      numeric: false, outcome: true },
  { key: 'result_margin',         label: 'Margin',     numeric: true,  outcome: true },
  { key: 'result_upset',          label: 'Upset',      numeric: false, outcome: true,
    get: r => (r.result_upset == null ? null : (r.result_upset ? 'Upset' : '')) },
];

function signedPct(v) { return `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%`; }

function renderMatchupYearSelect() {
  const el = document.getElementById('matchup-year-select');
  if (!el || !matchupData) return;
  el.innerHTML = matchupData.years.map(y =>
    `<option value="${y}"${y === matchupYear ? ' selected' : ''}>${y}</option>`
  ).join('');
}

function setMatchupYear(year) { matchupYear = Number(year); renderMatchupTable(); }

function sortMatchupBy(column) {
  if (matchupSortColumn === column) {
    matchupSortDir = matchupSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    matchupSortColumn = column;
    matchupSortDir = 'asc';
  }
  renderMatchupTable();
}

function setMatchupSearch(query) { matchupSearchQuery = query.trim().toLowerCase(); renderMatchupTable(); }

function filterMatchupRows(rows, query) {
  if (!query) return rows;
  return rows.filter(r =>
    [r.fav, r.dog, r.round, r.region].some(v => (v || '').toLowerCase().includes(query))
  );
}

function renderMatchupTable() {
  if (!matchupData || matchupYear == null) return;
  const headEl = document.getElementById('matchup-table-head');
  const bodyEl = document.getElementById('matchup-table-body');
  if (!headEl || !bodyEl) return;

  const firstOutcome = MATCHUP_COLUMNS.findIndex(c => c.outcome);
  const cls = (col, i) =>
    `${col.numeric ? ' numeric' : ''}${col.outcome ? ' outcome' : ''}${i === firstOutcome ? ' outcome-start' : ''}`;

  headEl.innerHTML = `<tr>${MATCHUP_COLUMNS.map((col, i) => {
    const sortOn = col.sortKey || col.key;
    const active = sortOn === matchupSortColumn;
    const arrow = active ? (matchupSortDir === 'asc' ? ' ▲' : ' ▼') : '';
    return `<th class="sortable${active ? ' active' : ''}${cls(col, i)}"
                onclick="sortMatchupBy('${sortOn}')">${col.label}${arrow}</th>`;
  }).join('')}</tr>`;

  const yearRows = matchupData.matchups_by_year[String(matchupYear)] || [];
  const sorted = sortStatsRows(filterMatchupRows(yearRows, matchupSearchQuery), matchupSortColumn, matchupSortDir);

  if (sorted.length === 0) {
    bodyEl.innerHTML = `<tr><td colspan="${MATCHUP_COLUMNS.length}" class="stats-empty">No games match “${matchupSearchQuery}”.</td></tr>`;
    return;
  }

  bodyEl.innerHTML = sorted.map(row => `<tr${row.result_upset ? ' class="upset-row"' : ''}>${MATCHUP_COLUMNS.map((col, i) => {
    const v = col.get ? col.get(row) : row[col.key];
    const display = v == null ? '—' : (col.fmt ? col.fmt(v) : v);
    return `<td class="${cls(col, i).trim()}">${display}</td>`;
  }).join('')}</tr>`).join('');
}

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

function renderMlBacktest() {
  const d = mlBacktestData;
  if (!d) return;

  const prod = d.models.find(m => m.key === d.production_key);
  const base = d.models.find(m => m.key === d.baseline_key);
  if (!prod || !base) return;

  // ── Headline metric cards ──
  const cards = [
    {
      label: 'Brier score',
      value: num4(prod.brier),
      sub: prod.brier_ci ? `95% CI ${num4(prod.brier_ci[0])}–${num4(prod.brier_ci[1])}` : '',
      hint: 'lower is better',
    },
    {
      label: 'Skill vs seed baseline (BSS)',
      value: `${prod.bss >= 0 ? '+' : ''}${prod.bss.toFixed(3)}`,
      sub: `baseline Brier ${num4(base.brier)}`,
      hint: prod.bss > 0 ? 'modest positive skill' : 'no skill',
      tone: prod.bss > 0 ? 'good' : 'bad',
    },
    {
      label: 'Winner accuracy',
      value: btPct(prod.accuracy),
      sub: `seed baseline ${btPct(base.accuracy)}`,
      hint: `${((prod.accuracy - base.accuracy) * 100).toFixed(1)}pp vs baseline`,
      // A sub-1pp gap on 1,323 games is ~6 extra games called right — well
      // inside noise, so this is flagged as flat rather than as a win.
      tone: Math.abs(prod.accuracy - base.accuracy) < 0.01 ? 'flat' : 'good',
    },
    {
      label: 'Games evaluated',
      value: String(d.n_games),
      sub: `${d.years.length} tournaments, ${d.years[0]}–${d.years[d.years.length - 1]}`,
      hint: 'all out-of-sample',
    },
  ];

  document.getElementById('bt-headline').innerHTML = cards.map(c => `
    <div class="bt-card${c.tone ? ` bt-card-${c.tone}` : ''}">
      <div class="bt-card-label">${c.label}</div>
      <div class="bt-card-value">${c.value}</div>
      <div class="bt-card-sub">${c.sub}</div>
      <div class="bt-card-hint">${c.hint}</div>
    </div>
  `).join('');

  // ── The honest headline: skill is real but small, and accuracy is flat ──
  const accGapPP = (prod.accuracy - base.accuracy) * 100;
  const extraGames = Math.round((prod.accuracy - base.accuracy) * d.n_games);
  document.getElementById('bt-callout').innerHTML = `
    <p><strong>The short version:</strong> the model beats the seed baseline on Brier score by
    ${(prod.bss * 100).toFixed(1)}% (BSS ${prod.bss >= 0 ? '+' : ''}${prod.bss.toFixed(3)}) — a real but
    modest edge that comes almost entirely from <em>better-calibrated confidence</em>, not from calling more
    games correctly. It picks the winner in ${btPct(prod.accuracy)} of games against
    ${btPct(base.accuracy)} for seeds alone: a gap of ${accGapPP.toFixed(1)} percentage points, or about
    ${extraGames} extra games out of ${d.n_games} across ${d.years.length} tournaments — small enough that
    it is indistinguishable from noise. If you want a model that tells you who wins substantially more often
    than the bracket's own seeding does, this is not that model, and neither was any of the more complex
    architectures tested.</p>
  `;

  const favEl = document.getElementById('bt-favrate');
  if (favEl) favEl.textContent = btPct(d.favorite_win_rate);
  const favEl2 = document.getElementById('bt-favrate-2');
  if (favEl2) favEl2.textContent = btPct(d.favorite_win_rate);
  const storedEl = document.getElementById('bt-stored-rate');
  if (storedEl && d.source_orientation_note) {
    storedEl.textContent = btPct(d.source_orientation_note.stored_outcome_1_rate);
  }

  // ── Model comparison table ──
  const mCols = ['Model', 'Games', 'Brier', '95% CI', 'BSS vs seed', 'Accuracy', 'Log loss'];
  document.getElementById('bt-models-head').innerHTML =
    `<tr>${mCols.map((c, i) => `<th class="${i ? 'numeric' : ''}">${c}</th>`).join('')}</tr>`;
  document.getElementById('bt-models-body').innerHTML = d.models.map(m => {
    const isBase = m.key === d.baseline_key;
    const isProd = m.key === d.production_key;
    const bssCls = isBase ? '' : (m.bss > 0 ? 'bt-pos' : 'bt-neg');
    const bssTxt = isBase ? '— (baseline)' : `${m.bss >= 0 ? '+' : ''}${m.bss.toFixed(4)}`;
    return `<tr${isProd ? ' class="bt-row-prod"' : ''}>
      <td>${m.label}${isProd ? ' <span class="bt-tag">production</span>' : ''}</td>
      <td class="numeric">${m.n_games}</td>
      <td class="numeric">${num4(m.brier)}</td>
      <td class="numeric bt-dim">${m.brier_ci ? `${num4(m.brier_ci[0])}–${num4(m.brier_ci[1])}` : '—'}</td>
      <td class="numeric ${bssCls}">${bssTxt}</td>
      <td class="numeric">${btPct(m.accuracy)}</td>
      <td class="numeric">${num4(m.log_loss)}</td>
    </tr>`;
  }).join('');

  // ── Per-year skill chart (diverging bars around zero) ──
  const maxAbs = Math.max(...d.per_year.map(y => Math.abs(y.bss)), 0.01);
  document.getElementById('bt-year-chart').innerHTML = d.per_year.map(y => {
    const h = Math.max(3, Math.round((Math.abs(y.bss) / maxAbs) * 46));
    const pos = y.bss >= 0;
    return `
      <div class="bt-year" title="${y.year}: BSS ${y.bss >= 0 ? '+' : ''}${y.bss.toFixed(3)} — model Brier ${num4(y.brier_model)} vs seed ${num4(y.brier_seed)} (${y.n_games} games)">
        <div class="bt-year-up">${pos ? `<div class="bt-year-bar bt-year-pos" style="height:${h}px"></div>` : ''}</div>
        <div class="bt-year-axis"></div>
        <div class="bt-year-dn">${pos ? '' : `<div class="bt-year-bar bt-year-neg" style="height:${h}px"></div>`}</div>
        <div class="bt-year-label">'${String(y.year).slice(2)}</div>
      </div>`;
  }).join('');

  const losses = d.per_year.filter(y => y.bss < 0);
  document.getElementById('bt-year-note').textContent =
    `The model lost to the seed baseline in ${losses.length} of ${d.per_year.length} tournaments` +
    (losses.length ? ` (${losses.map(y => y.year).join(', ')}).` : '.') +
    ' Single-year swings are dominated by a handful of games — 63 per tournament.';

  // ── Per-round table ──
  const rCols = ['Round', 'Games', 'Upset rate', 'Brier', 'Seed Brier', 'BSS', 'Accuracy', 'Seed accuracy'];
  document.getElementById('bt-rounds-head').innerHTML =
    `<tr>${rCols.map((c, i) => `<th class="${i ? 'numeric' : ''}">${c}</th>`).join('')}</tr>`;
  document.getElementById('bt-rounds-body').innerHTML = d.per_round.map(r => {
    const accWorse = r.accuracy_model < r.accuracy_seed - 1e-9;
    return `<tr>
      <td>${r.label}</td>
      <td class="numeric">${r.n_games}</td>
      <td class="numeric">${btPct(r.upset_rate)}</td>
      <td class="numeric">${num4(r.brier_model)}</td>
      <td class="numeric bt-dim">${num4(r.brier_seed)}</td>
      <td class="numeric ${r.bss > 0 ? 'bt-pos' : 'bt-neg'}">${r.bss >= 0 ? '+' : ''}${r.bss.toFixed(4)}</td>
      <td class="numeric ${accWorse ? 'bt-neg' : ''}">${btPct(r.accuracy_model)}</td>
      <td class="numeric bt-dim">${btPct(r.accuracy_seed)}</td>
    </tr>`;
  }).join('');

  // ── Market comparison ──
  const mk = d.market_subset;
  const mkSection = document.getElementById('bt-market-section');
  if (!mk) {
    mkSection.style.display = 'none';
  } else {
    const beatsMarket = mk.model.brier < mk.market.brier;
    document.getElementById('bt-market-intro').innerHTML =
      `Closing betting odds exist for ${mk.n_games} of ${d.n_games} games (${mk.years[0]}–${mk.years[mk.years.length - 1]}),
       so this is scored on that subset only and is not comparable to the full-sample numbers above.
       The market is the honest benchmark for a forecasting model — it aggregates every public signal plus money.
       <strong>${beatsMarket
         ? 'On this subset the model edges the closing line.'
         : 'On this subset the closing line is still slightly sharper than the model.'}</strong>`;

    const rows = [
      { label: 'Torvik ratings (production)', s: mk.model },
      { label: 'Closing betting market', s: mk.market },
      { label: 'Seed baseline', s: mk.seed },
    ];
    document.getElementById('bt-market-head').innerHTML =
      `<tr><th>Source</th><th class="numeric">Brier</th><th class="numeric">BSS vs seed</th><th class="numeric">Accuracy</th><th class="numeric">Log loss</th></tr>`;
    document.getElementById('bt-market-body').innerHTML = rows.map(r => `
      <tr>
        <td>${r.label}</td>
        <td class="numeric">${num4(r.s.brier)}</td>
        <td class="numeric">${r.s.bss == null ? '— (baseline)' : `${r.s.bss >= 0 ? '+' : ''}${r.s.bss.toFixed(4)}`}</td>
        <td class="numeric">${btPct(r.s.accuracy)}</td>
        <td class="numeric">${num4(r.s.log_loss)}</td>
      </tr>`).join('');
  }

  // ── Calibration ──
  const bins = d.calibration.bins.filter(b => b.count > 0);
  document.getElementById('bt-calib').innerHTML = bins.map(b => {
    const predH = Math.round(b.mean_predicted * 80);
    const actH  = Math.round(b.mean_actual * 80);
    return `
      <div class="bt-calib-bin" title="Predicted ${btPct(b.mean_predicted)} → actual ${btPct(b.mean_actual)} (${b.count} games)">
        <div class="bt-calib-bars">
          <div class="bt-calib-bar bt-calib-pred" style="height:${predH}px"></div>
          <div class="bt-calib-bar bt-calib-act"  style="height:${actH}px"></div>
        </div>
        <div class="bt-calib-x">${Math.round(b.lower * 100)}–${Math.round(b.upper * 100)}</div>
        <div class="bt-calib-n">n=${b.count}</div>
      </div>`;
  }).join('');

  document.getElementById('bt-calib-note').innerHTML =
    `<span class="bt-swatch bt-calib-pred"></span> predicted &nbsp;
     <span class="bt-swatch bt-calib-act"></span> actual &nbsp;·&nbsp;
     Expected calibration error <strong>${num4(d.calibration.ece)}</strong> —
     population-weighted mean gap between predicted and actual. Computed over ${d.n_games} games in
     10 buckets, several of which are thin enough that their gap is mostly sampling noise.`;
}
