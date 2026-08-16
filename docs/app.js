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
      return effectiveBarthag(t1) >= effectiveBarthag(t2) ? t1 : t2;   // tie-break
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

// bracketData[approach][base] — one precomputed bracket JSON per (approach,
// probability base) combination. approach ∈ {pool, exhaustive, stat}; each
// was actually constructed under that base (see scripts/prob_base_variants.py
// + generate_{poolaware,exhaustive,region}_bracket.py --prob-base), so
// switching bases genuinely changes picks, not just displayed odds. A base
// that failed to fetch (or was never generated) stays null and callers fall
// back to torvik.
let bracketData = { pool: {}, exhaustive: {}, stat: {} };
let loyoData          = null;   // per-year ESPN points, see docs/data/loyo_points.json
let factorsData        = null;  // elo/ap barthag, see docs/data/team_factors.json
let actualData         = null;  // real 2026 outcome, see docs/data/actual_2026.json — the
                                 // 2026 tournament already concluded, this is a replay
let window3yrData      = null;  // 2024-2026 backtest window, see docs/data/loyo_window_3yr.json
let currentWindow      = '15yr';   // '15yr' | '3yr' — which backtest window's P(1st)/note to display
let currentProbBase   = 'torvik';  // 'torvik' | 'elo' | 'ap' | 'upset' — applies to every approach
let teamIndex        = {};      // team_id → { barthag, elo_barthag, ap_barthag, adj_oe, adj_de, champ_prob, elo_rating }
let currentKey       = 'pool';
let currentRound  = 'Round of 64';
let roundsCache   = {};         // "key_base" → computed rounds[]

const BRACKET_FILES = {
  pool:       { torvik: 'bracket_2026.json',           elo: 'bracket_2026_elo.json',           ap: 'bracket_2026_ap.json',           upset: 'bracket_2026_upset.json' },
  exhaustive: { torvik: 'bracket_2026_exhaustive.json', elo: 'bracket_2026_exhaustive_elo.json', ap: 'bracket_2026_exhaustive_ap.json', upset: 'bracket_2026_exhaustive_upset.json' },
  stat:       { torvik: 'bracket_2026_region.json',     elo: 'bracket_2026_region_elo.json',     ap: 'bracket_2026_region_ap.json',     upset: 'bracket_2026_region_upset.json' },
};

// ──────────────────────────────────────────────────────────────────
// BOOT
// ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  let profiles;
  try {
    const [poolTv, exhaustiveTv, regionTv, profilesRes] = await Promise.all([
      fetch('data/bracket_2026.json?v=2026-08-16c').then(r => r.json()),
      fetch('data/bracket_2026_exhaustive.json?v=2026-08-16c').then(r => r.json()),
      fetch('data/bracket_2026_region.json?v=2026-08-16c').then(r => r.json()),
      fetch('data/team_profiles.json?v=2026-08-16c').then(r => r.json()),
    ]);
    bracketData.pool.torvik       = poolTv;
    bracketData.exhaustive.torvik = exhaustiveTv;
    bracketData.stat.torvik       = regionTv;
    profiles = profilesRes;
  } catch (err) {
    document.body.innerHTML =
      '<p style="padding:48px;font-family:sans-serif;color:#bb4d2d">Failed to load bracket data. ' +
      'Make sure bracket_2026.json, bracket_2026_exhaustive.json, bracket_2026_region.json, ' +
      'and team_profiles.json are present in docs/data/.</p>';
    return;
  }

  // Elo/AP/Upset probability-base variants and the per-year points/factors
  // panels are all nice-to-haves, not required to render the bracket picker
  // — fetch separately so a missing/broken file degrades gracefully (falls
  // back to torvik for a bracket variant, "no panel" for loyo/factors)
  // instead of blocking the whole page.
  const altFetches = [];
  for (const approach of ['pool', 'exhaustive', 'stat']) {
    for (const base of ['elo', 'ap', 'upset']) {
      altFetches.push(
        fetch(`data/${BRACKET_FILES[approach][base]}?v=2026-08-16c`)
          .then(r => r.json())
          .then(data => { bracketData[approach][base] = data; })
          .catch(() => { bracketData[approach][base] = null; })
      );
    }
  }
  await Promise.all(altFetches);

  let loyo, factors;
  try {
    loyo = await fetch('data/loyo_points.json?v=2026-08-16c').then(r => r.json());
  } catch (err) {
    loyo = null;
  }
  try {
    factors = await fetch('data/team_factors.json?v=2026-08-16c').then(r => r.json());
  } catch (err) {
    factors = null;
  }
  let actual;
  try {
    actual = await fetch('data/actual_2026.json?v=2026-08-16c').then(r => r.json());
  } catch (err) {
    actual = null;
  }
  let window3yr;
  try {
    window3yr = await fetch('data/loyo_window_3yr.json?v=2026-08-16c').then(r => r.json());
  } catch (err) {
    window3yr = null;
  }

  loyoData       = loyo;
  factorsData    = factors;
  actualData     = actual;
  window3yrData  = window3yr;

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
  // Merge in the elo/ap barthag lenses (docs/data/team_factors.json), if
  // the fetch above succeeded. Missing entries just fall back to torvik
  // barthag in effectiveBarthag() — never a hard failure. Only Chalk (no
  // precomputed bracket per base) uses this directly; pool/exhaustive/stat
  // get their alt-base picks from the precomputed JSON above instead.
  if (factorsData && factorsData.teams) {
    for (const t of factorsData.teams) {
      if (teamIndex[t.team_id]) {
        teamIndex[t.team_id].elo_barthag = t.elo_barthag;
        teamIndex[t.team_id].ap_barthag  = t.ap_barthag;
      }
    }
  }

  renderStrategyStrip();
  renderProbBaseToggle();
  renderWindowToggle();
  activateStrategy('pool');
});

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
    elo_barthag:    prof.elo_barthag    ?? null,
    ap_barthag:     prof.ap_barthag     ?? null,
    adj_oe:         prof.adj_oe         ?? null,
    adj_de:         prof.adj_de         ?? null,
    champ_prob:     prof.champ_prob     ?? null,
    elo_rating:     prof.elo_rating     ?? null,
  };
}

// The barthag value that drives Chalk's live win probabilities under the
// selected global probability base (see renderProbBaseToggle). Falls back
// to torvik barthag when a team has no data for the requested lens — never
// a hard failure. Only Chalk uses this: pool/exhaustive/stat get their
// alt-base picks from a genuinely separate precomputed bracket instead
// (see bracketData / BRACKET_FILES), not a client-side recompute.
function effectiveBarthag(team) {
  if (currentProbBase === 'torvik') return team.barthag;
  if (currentProbBase === 'ap') return team.ap_barthag ?? team.barthag;
  // 'elo' and 'upset' share the same underlying rating (see
  // scripts/prob_base_variants.py's UNDERLYING_BASE) — Chalk has no
  // risk_level concept (seed-first, probability only breaks same-seed
  // ties), so "upset" can't mean anything different for Chalk than elo.
  return team.elo_barthag ?? team.barthag;
}

// Convert a pre-computed bracket JSON into the internal game format. Picks
// AND win_prob both come straight from the JSON — it was actually
// constructed under the selected probability base (see BRACKET_FILES),
// not recomputed client-side.
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

// Each precomputed approach falls back to its torvik bracket if the
// selected base failed to load (e.g. generation skipped for that year).
function poolRounds()       { return precomputedRounds(bracketData.pool[currentProbBase]       ?? bracketData.pool.torvik); }
function exhaustiveRounds() { return precomputedRounds(bracketData.exhaustive[currentProbBase]  ?? bracketData.exhaustive.torvik); }
function regionRounds()     { return precomputedRounds(bracketData.stat[currentProbBase]        ?? bracketData.stat.torvik); }

// Precomputed strategies read their bracket straight from a JSON file
// instead of simulating client-side (see STRATEGIES pick === null).
const PRECOMPUTED_ROUNDS = {
  pool: poolRounds,
  exhaustive: exhaustiveRounds,
  stat: regionRounds,
};

// Simulate the full bracket for Chalk (the only strategy without a
// precomputed bracket per probability base) from R64. R64 matchup
// structure (which teams play which) is identical across every approach
// and base — only pulled from bracketData.pool.torvik for convenience —
// but win_prob is recomputed via effectiveBarthag() under the currently
// selected base, so Chalk's same-seed tie-breaks genuinely follow it too.
function simulate(strategy) {
  const r64 = bracketData.pool.torvik.rounds[0].games.map(g => {
    const team1 = mkTeam(g.team1_id, g.team1, g.team1_seed, g.team1_rating);
    const team2 = mkTeam(g.team2_id, g.team2, g.team2_seed, g.team2_rating);
    const win_prob = currentProbBase === 'torvik' ? g.win_prob : log5(effectiveBarthag(team1), effectiveBarthag(team2));
    return {
      round:    'Round of 64',
      region:   g.region,
      team1,
      team2,
      win_prob,
      is_upset: currentProbBase === 'torvik' ? g.is_upset : upsetCheck(team1, team2, win_prob),
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
        const wp = log5(effectiveBarthag(w1), effectiveBarthag(w2));
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
  const wp = log5(effectiveBarthag(t1), effectiveBarthag(t2));
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
  return effectiveBarthag(game.team1) >= effectiveBarthag(game.team2) ? game.team1 : game.team2;
}

// Get cached rounds for a strategy key. Cache key includes the probability
// base — a precomputed strategy's rounds genuinely differ per base now
// (different source JSON), not just Chalk.
function getRounds(key) {
  const cacheKey = `${key}_${currentProbBase}`;
  if (roundsCache[cacheKey]) return roundsCache[cacheKey];
  const s = STRATEGIES.find(s => s.key === key);
  if (s.pick === null) {
    roundsCache[cacheKey] = PRECOMPUTED_ROUNDS[key]();
  } else {
    roundsCache[cacheKey] = simulate(s);
  }
  return roundsCache[cacheKey];
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
// window3yrData (docs/data/loyo_window_3yr.json) holds a second, much
// lower-power (n=3) cut over just 2024-2026 for comparing strategies on
// the recent "meta" specifically — diagnostic only, too few years for the
// paired significance tests the 15-yr view runs (see the note text). Only
// swaps the displayed P(1st)/badge/note; picks themselves are unaffected —
// the backtest window doesn't change which bracket a strategy actually built.
const WINDOW_DEFS = {
  '15yr': { label: '15-Year (2011–2026)', note: 'Full backtest, N=31 pool. Statistically validated.' },
  '3yr':  { label: '2024–2026 (3-yr)', note: 'Diagnostic only — too few years for significance testing.' },
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
// 15-yr numbers if the 3-yr window has no entry for it (e.g. Chalk, which
// isn't backtested as a standalone strategy in either window).
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
  pool:       () => bracketData.pool[currentProbBase]       ?? bracketData.pool.torvik,
  exhaustive: () => bracketData.exhaustive[currentProbBase] ?? bracketData.exhaustive.torvik,
  stat:       () => bracketData.stat[currentProbBase]       ?? bracketData.stat.torvik,
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
    return `
      <div class="loyo-chip" title="${y}: ${v.toFixed(0)} pts">
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
// PROBABILITY BASE TOGGLE
//
// Applies to every approach, not just Chalk: Pool Optimizer / Exhaustive
// Search / Region Beam Search each have a real precomputed bracket per
// base (see BRACKET_FILES / scripts/prob_base_variants.py) — the same
// construction algorithm actually run against elo/ap_strength round_probs
// instead of torvik, so switching bases can change which team a given
// approach picks, not just what percentage is shown. elo/ap are fully
// independent rating systems (not derived from torvik at all), unlike
// the roster_adj/coach_adj lens this replaced 2026-08-15 — those rarely
// moved a single pick for the 2026 field (small, capped adjustments to
// Torvik's own round_probs). elo/ap disagree with Torvik on 20/68 and
// 14/68 first-round favorites respectively; neither is separately
// backtested as a standalone P(1st) strategy.
//
// "upset" (added same day): even elo/ap mostly kept Duke/Michigan/
// Arizona in the Final Four — reasonable rating systems agree on who's
// genuinely elite. Real Final Four variation needed risk_level (bracket_
// construction.py's contrarian-weighting knob) pushed to max, on top of
// elo's round_probs — verified against the real 2026 field: zero 1-seeds
// in the Final Four, Miami (OH) as champion under Pool Optimizer/Region.
// For Pool Optimizer specifically this bypasses the normal pool-
// simulation candidate selection (see generate_poolaware_bracket.py) —
// that selection correctly rejects near-0%-real-odds picks, which is
// WHY it's the validated strategy, so "upset" is a direct, single-shot,
// explicitly unvalidated "what if" construction instead.
// ──────────────────────────────────────────────────────────────────

const PROB_BASE_DEFS = {
  torvik: { label: 'Torvik', note: 'The backtested base every approach is measured against.' },
  elo:    { label: 'Elo Rating', note: 'Independent Elo rating. Disagrees with Torvik on 20/68 R1 favorites. Not separately backtested.' },
  ap:     { label: 'AP Poll Strength', note: 'Human-voter poll, not efficiency stats. Disagrees with Torvik on 14/68 R1 favorites. Not separately backtested.' },
  upset:  { label: 'Upset Hunter', note: 'Max-contrarian Elo weighting. Zero 1-seeds in its Final Four. Unvalidated "what if" build.' },
};

function setProbBase(base) {
  currentProbBase = base;
  renderProbBaseToggle();
  activateStrategy(currentKey);
}

function renderProbBaseToggle() {
  const toggleEl = document.getElementById('probbase-toggle');
  const noteEl   = document.getElementById('probbase-note');
  if (!toggleEl) return;

  toggleEl.innerHTML = Object.entries(PROB_BASE_DEFS).map(([key, def]) => `
    <button class="probbase-toggle-btn${key === currentProbBase ? ' active' : ''}"
            onclick="setProbBase('${key}')">${def.label}</button>
  `).join('');

  if (noteEl) noteEl.textContent = PROB_BASE_DEFS[currentProbBase].note;
}

