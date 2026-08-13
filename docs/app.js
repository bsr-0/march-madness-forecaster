/* ══════════════════════════════════════════════════════════════════
 * March Madness Forecaster — Bracket Picker
 * ══════════════════════════════════════════════════════════════════ */

// ──────────────────────────────────────────────────────────────────
// STRATEGY CATALOG
//
// This is the single source of truth for what appears in the
// strategy selector. Keep it accurate — it directly drives the UI.
//
// ═══ CURRENT BEST MODELS (2026-05-25) ═══════════════════════════
//   Best:   meta_region_poolaware — 11.9% P(1st), 14-yr LOYO, p=0.008
//   Strong: meta_region           —  8.0% P(1st), 14-yr LOYO, p<0.001
//   Strong: meta_exhaustive        —  7.7% P(1st), 14-yr LOYO, p<0.001
//   Seed baseline (random-ish):      3.1% P(1st)
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
      'Generates ~25 diverse candidate brackets and selects the one with the highest simulated P(win) ' +
      'against a realistic pool field. The strongest strategy across 14 years of historical backtests — ' +
      'outperforms the seed baseline every single year.',
    p_first: 11.9,          // ← update this after each backtest run
    badge: '11.9% P(1st)',
    badge_tone: 'gold',
    is_top: true,           // ← set to true for the current best model
    is_model: true,
    backtest_note:
      'Beats seed baseline 14/14 years. p = 0.008. ' +
      'Source: meta_region_poolaware, 14-yr LOYO backtest, N≈30 pool. ' +
      'Bracket: docs/data/bracket_2026.json.',
    // Pre-computed picks: pick === null means use winner_id from bracket_2026.json.
    // To swap to a new season's bracket, update the JSON file and re-deploy.
    pick: null,
  },
  {
    key: 'exhaustive',
    label: 'Exhaustive Search',
    subtitle: 'Best champion by simulation',
    description:
      'Tests all 64 possible champions and builds the full bracket that maximizes expected pool points ' +
      'for each. Picks whichever champion produces the highest-scoring bracket overall — more targeted ' +
      'than regional beam search. Validated via the same 14-year walk-forward backtest as Pool Optimizer.',
    p_first: 7.7,
    badge: '~7.7% P(1st)',
    badge_tone: 'green',
    is_top: false,
    is_model: true,
    backtest_note:
      'meta_exhaustive: 7.7% P(1st), 11/14 years, p<0.001. ' +
      'Source: 14-yr LOYO backtest, N≈30 pool. ' +
      'Bracket: docs/data/bracket_2026_exhaustive.json. Champion: Michigan (1-seed).',
    pick: null,
  },
  {
    key: 'stat',
    label: 'Region Beam Search',
    subtitle: 'Region-level construction',
    description:
      'Builds each region independently via beam search over Torvik round probabilities, then ' +
      'assembles the Final Four and champion from the region winners. The algorithm ' +
      'meta_region_poolaware is built on top of. Outperforms the seed baseline 11 out of 14 years.',
    p_first: 8.0,
    badge: '~8% P(1st)',
    badge_tone: 'green',
    is_top: false,
    is_model: true,
    backtest_note:
      'meta_region: 8.0% P(1st), 11/14 years, p<0.001. ' +
      'Source: 14-yr LOYO backtest, N≈30 pool. ' +
      'Bracket: docs/data/bracket_2026_region.json.',
    // Pre-computed picks: pick === null means use winner_id from bracket_2026_region.json.
    pick: null,
  },
  {
    key: 'chalk',
    label: 'Chalk',
    subtitle: 'Always the favorite',
    description:
      'Pick the lower seed (stronger team) in every game. Ties broken by Barthag rating. ' +
      'Simple and safe — but historically equivalent to the seed baseline (~3% P(1st)) ' +
      'and offers no differentiation in winner-take-all pools.',
    p_first: null,
    badge: 'Traditional',
    badge_tone: 'neutral',
    is_top: false,
    is_model: false,
    backtest_note:
      'Similar to seed baseline (~3.1% P(1st)). No upside differentiation in winner-take-all pools. ' +
      'Use this if you want a conventional, defensible bracket.',
    pick: (t1, t2) => {
      if (t1.seed !== t2.seed) return t1.seed < t2.seed ? t1 : t2;
      return t1.barthag >= t2.barthag ? t1 : t2;   // tie-break
    },
  },
  {
    key: 'upset',
    label: 'Upset Year',
    subtitle: 'Lean into the chaos',
    description:
      'Pick the underdog whenever the model gives them ≥ 38% probability — otherwise falls back to chalk. ' +
      'Captures the 5/12, 10/7, and close 8/9 games where chaos is plausible. ' +
      'Good for upset-heavy years like 2023 where favorites flamed out early.',
    p_first: null,
    badge: 'High Risk',
    badge_tone: 'red',
    is_top: false,
    is_model: false,
    backtest_note:
      'No systematic backtest. Higher ceiling, higher variance. ' +
      'High variance — good for years where chalk flops early. No systematic backtest.',
    pick: (t1, t2, winProb) => {
      // winProb is t1's probability. Underdog = higher seed number.
      const t1IsUnderdog = t1.seed > t2.seed;
      const underdogProb = t1IsUnderdog ? winProb : 1 - winProb;
      if (underdogProb >= 0.38) return t1IsUnderdog ? t1 : t2;
      // chalk fallback for non-competitive underdogs
      if (t1.seed !== t2.seed) return t1.seed < t2.seed ? t1 : t2;
      return t1.barthag >= t2.barthag ? t1 : t2;
    },
  },
  {
    key: 'champ_odds',
    label: 'Championship Odds',
    subtitle: 'Path-adjusted win probability',
    description:
      'Picks the team with the higher pre-tournament championship probability in each matchup. ' +
      'Unlike raw efficiency ratings, this accounts for bracket draw — a 1-seed facing a tough ' +
      'region gets a lower probability than one in an easier half. Useful for surfacing picks ' +
      'where path difficulty makes a meaningful difference.',
    p_first: null,
    badge: 'Stat Lens',
    badge_tone: 'sky',
    is_top: false,
    is_model: false,
    backtest_note:
      'No systematic pool backtest. Championship probability accounts for bracket path difficulty ' +
      'in addition to raw ability — differs from Barthag when draw is uneven.',
    pick: (t1, t2) => (t1.champ_prob ?? 0) >= (t2.champ_prob ?? 0) ? t1 : t2,
  },
  {
    key: 'offense',
    label: 'Offensive Edge',
    subtitle: 'Favor explosive offenses',
    description:
      'Pick the team with the higher adjusted offensive efficiency. ' +
      'Favors high-scoring, fast-paced teams. Useful as a lens for individual close games ' +
      'in years where tempo and scoring output dominate tournament outcomes.',
    p_first: null,
    badge: 'Stat Lens',
    badge_tone: 'sky',
    is_top: false,
    is_model: false,
    backtest_note:
      'No systematic pool backtest. Use as a supplemental stat lens, not a primary strategy.',
    pick: (t1, t2) => (t1.adj_oe ?? 100) >= (t2.adj_oe ?? 100) ? t1 : t2,
  },
  {
    key: 'defense',
    label: 'Defensive Wall',
    subtitle: 'Trust stingy defenses',
    description:
      'Pick the team with the lower adjusted defensive efficiency (harder to score against). ' +
      'Tournament basketball often rewards teams that suffocate opponents — ' +
      'particularly in high-leverage late rounds.',
    p_first: null,
    badge: 'Stat Lens',
    badge_tone: 'sky',
    is_top: false,
    is_model: false,
    backtest_note:
      'No systematic pool backtest. Use as a supplemental stat lens, not a primary strategy.',
    pick: (t1, t2) => (t1.adj_de ?? 120) <= (t2.adj_de ?? 120) ? t1 : t2,
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

// Colors per strategy badge tone (inline styles avoid needing a CSS class per strategy)
const BADGE_COLORS = {
  gold:    { bg: 'rgba(200,145,37,0.18)',  text: '#6b4800' },
  green:   { bg: 'rgba(47,122,99,0.16)',   text: '#1d5940' },
  red:     { bg: 'rgba(187,77,45,0.16)',   text: '#6e2211' },
  sky:     { bg: 'rgba(47,94,138,0.14)',   text: '#20405e' },
  neutral: { bg: 'rgba(92,64,41,0.10)',    text: '#4f3720' },
};

// ── App state ──

let bracketData      = null;
let exhaustiveData   = null;
let regionData        = null;
let teamIndex        = {};      // team_id → { barthag, adj_oe, adj_de, champ_prob, elo_rating }
let currentKey       = 'pool';
let currentRound  = 'Round of 64';
let roundsCache   = {};         // strategy key → computed rounds[]

// ──────────────────────────────────────────────────────────────────
// BOOT
// ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  let bracket, exhaustive, region, profiles;
  try {
    [bracket, exhaustive, region, profiles] = await Promise.all([
      fetch('data/bracket_2026.json?v=2026-08-13').then(r => r.json()),
      fetch('data/bracket_2026_exhaustive.json?v=2026-08-13').then(r => r.json()),
      fetch('data/bracket_2026_region.json?v=2026-08-13').then(r => r.json()),
      fetch('data/team_profiles.json?v=2026-08-13').then(r => r.json()),
    ]);
  } catch (err) {
    document.body.innerHTML =
      '<p style="padding:48px;font-family:sans-serif;color:#bb4d2d">Failed to load bracket data. ' +
      'Make sure bracket_2026.json, bracket_2026_exhaustive.json, bracket_2026_region.json, ' +
      'and team_profiles.json are present in docs/data/.</p>';
    return;
  }

  bracketData    = bracket;
  exhaustiveData = exhaustive;
  regionData     = region;

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
    name:       rawName.replace(/^\(\d+\)\s*/, ''),
    seed,
    barthag:    prof.barthag    ?? ratingFallback,
    adj_oe:     prof.adj_oe     ?? null,
    adj_de:     prof.adj_de     ?? null,
    champ_prob: prof.champ_prob ?? null,
    elo_rating: prof.elo_rating ?? null,
  };
}

// Convert a pre-computed bracket JSON into the internal game format.
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

function poolRounds()       { return precomputedRounds(bracketData); }
function exhaustiveRounds() { return precomputedRounds(exhaustiveData); }
function regionRounds()     { return precomputedRounds(regionData); }

// Precomputed strategies read their bracket straight from a JSON file
// instead of simulating client-side (see STRATEGIES pick === null).
const PRECOMPUTED_ROUNDS = {
  pool: poolRounds,
  exhaustive: exhaustiveRounds,
  stat: regionRounds,
};

// Simulate the full bracket for a non-pool strategy from R64.
function simulate(strategy) {
  const r64 = bracketData.rounds[0].games.map(g => ({
    round:    'Round of 64',
    region:   g.region,
    team1:    mkTeam(g.team1_id, g.team1, g.team1_seed, g.team1_rating),
    team2:    mkTeam(g.team2_id, g.team2, g.team2_seed, g.team2_rating),
    win_prob: g.win_prob,     // use model's R64 probabilities
    is_upset: g.is_upset,
  }));

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
  renderRoundTabs(rounds);
  renderGames(rounds);
}

// ── Strategy strip ──

function renderStrategyStrip() {
  const el = document.getElementById('strategy-strip');
  el.innerHTML = STRATEGIES.map(s => {
    const bc = BADGE_COLORS[s.badge_tone] || BADGE_COLORS.neutral;
    const badgeHTML = s.badge
      ? `<span class="strategy-badge" style="background:${bc.bg};color:${bc.text}">${s.badge}</span>`
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
const OPPONENT_SOURCE_DATA = { pool: () => bracketData, exhaustive: () => exhaustiveData, stat: () => regionData };

function renderStrategyDetail() {
  const el = document.getElementById('strategy-detail');
  const s  = STRATEGIES.find(s => s.key === currentKey);
  const bc = BADGE_COLORS[s.badge_tone] || BADGE_COLORS.neutral;

  const tagHTML = s.is_model
    ? `<span class="sd-tag sd-tag-model">Model-backed</span>`
    : `<span class="sd-tag sd-tag-pref">Preference lens</span>`;

  const perfHTML = s.p_first != null
    ? `<span class="sd-perf" style="background:${bc.bg};color:${bc.text}">${s.p_first}% P(1st) historically</span>`
    : '';

  const dataFn = OPPONENT_SOURCE_DATA[currentKey];
  const opponentSource = dataFn ? dataFn()?.opponent_source : null;
  const opponentHTML = opponentSource
    ? `<p class="sd-opponent">"% of pool" on each pick below is from <strong>${opponentSource}</strong> — ` +
      `what your actual opponents picked, independent of which model built this bracket.</p>`
    : '';

  el.innerHTML = `
    <div class="sd-row">${tagHTML}${perfHTML}</div>
    <p class="sd-desc">${s.description}</p>
    <p class="sd-note">${s.backtest_note}</p>
    ${opponentHTML}
  `;
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

  return `
    <div class="game-card${isUpsetPick ? ' upset' : ''}">
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
