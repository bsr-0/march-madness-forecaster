/* Checks the pick exporter in docs/app.js.
 *
 * The job this page exists to finish is 63 picks typed into a pool site. Until
 * 2026-09-06 there was no way to get them off the page at all: a user read a
 * six-column horizontally-scrolling board -- on a phone, one column at a time
 * -- and retyped it. Every modelling decision in this repo sits upstream of
 * that step, so an exporter that quietly drops or mislabels picks costs more
 * than a model that is slightly wrong.
 *
 * The first version of this exporter looked right and was wrong: it resolved
 * winners by team id when g.win is an INDEX into season.teams, so every line
 * read "undefined 8". It rendered, it did not throw, and the bug was only
 * visible by reading the output. Hence checking the output.
 *
 * app.js is browser-global rather than a module, so it is evaluated in a vm
 * with the smallest stubs that let it load. init() runs on load and awaits a
 * fetch that never resolves, which is exactly what we want: no DOM is touched.
 *
 * Run: node tests/test_picks_export.js
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
    console.log('  ok   ' + name);
  } catch (e) {
    console.error('  FAIL ' + name + '\n       ' + e.message);
    process.exitCode = 1;
  }
}

const ctxHash = { value: '' };

function loadApp(hash) {
  // fit.js first: app.js calls fitLinear/winProb/bracketAdvancementProbs etc.
  // as bare globals, exactly as the real page loads it via a preceding
  // <script> tag (see index.html) rather than a module import.
  const fitSrc = fs.readFileSync(path.join(__dirname, '..', 'docs', 'fit.js'), 'utf8');
  const src = fs.readFileSync(path.join(__dirname, '..', 'docs', 'app.js'), 'utf8');
  const noop = () => {};
  const ctx = {
    console,
    setTimeout,
    clearTimeout,
    // init() awaits this forever, so it never reaches the DOM.
    fetch: () => new Promise(() => {}),
    document: {
      getElementById: () => null,
      querySelector: () => null,
      querySelectorAll: () => [],
      addEventListener: noop,
    },
    window: { isSecureContext: false },
    navigator: {},
    URLSearchParams,  // global in browsers, needs passing into the vm
    location: { hash: '' },
    history: { replaceState: (a, b, url) => { ctxHash.value = url; } },
    module: undefined,  // fit.js only attaches to module.exports if this exists
  };
  if (hash) ctx.location.hash = hash;
  vm.createContext(ctx);
  vm.runInContext(fitSrc, ctx);
  vm.runInContext(src, ctx);
  // Top-level `const` lives in the script's lexical scope, not on the context
  // object, so reach it by evaluating in that same scope.
  vm.runInContext(
    'globalThis.__api = { state, picksAsText, ROUNDS, readHash, writeHash, CUSTOM, MODEL, solveFromPicks, '
    + 'pickDefaultSeason, p1Pct, refit, percentileInField, ordinal, fittedEval, solveByFit };', ctx);
  return ctx.__api;
}

/* A four-team season is enough: the exporter walks whatever rounds it is given,
 * so the shape matters and the size does not. */
function fixture(app) {
  app.state.season = {
    teams: [
      { id: 'duke', name: 'Duke', seed: 1 },
      { id: 'akron', name: 'Akron', seed: 16 },
      { id: 'iowa', name: 'Iowa', seed: 8 },
      { id: 'utah', name: 'Utah', seed: 9 },
    ],
    strategies: [{ id: 'p1', label: 'Maximise chance of winning', p1: 0.099, ev: 874 }],
    p1_assumption: 'P(1st) assumes a 30-opponent pool with ESPN public pick behaviour.',
  };
  app.state.year = 2027;
  app.state.strategy = 'p1';
  app.state.rounds = [
    [{ a: 0, b: 1, win: 0 }, { a: 2, b: 3, win: 3 }],
    [{ a: 0, b: 3, win: 0 }],
  ];
  return app;
}

console.log('pick export');

check('every winner is named, none undefined', () => {
  const app = fixture(loadApp());
  const text = app.picksAsText();
  assert.ok(!/undefined/.test(text), 'export contains "undefined":\n' + text);
  assert.ok(text.includes('1 Duke'), 'missing the round-1 winner');
  assert.ok(text.includes('9 Utah'), 'missing the upset winner');
});

check('winners are resolved by index, not by id', () => {
  // The exact defect that shipped: teams.find(x => x.id === g.win) returns
  // undefined for every game, because g.win is an index.
  const app = fixture(loadApp());
  const text = app.picksAsText();
  assert.ok(!text.includes('9 Iowa'), 'resolved the wrong side of the game');
  assert.ok(text.includes('9 Utah'), 'index 3 must resolve to Utah, seed 9');
});

check('one line per game, in bracket order', () => {
  const app = fixture(loadApp());
  const lines = app.picksAsText().split('\n').filter(l => /^ {2}\d+ \S/.test(l));
  assert.strictEqual(lines.length, 3, 'expected 3 winners for 2+1 games');
  assert.ok(lines[2].includes('Duke'), 'the champion must come last');
});

check('the disclosure travels with the picks', () => {
  // A bracket pasted into a group chat outlives the page it came from, and the
  // number goes with it. This is the mandatory product.v3 disclosure.
  const app = fixture(loadApp());
  assert.ok(/30-opponent pool/.test(app.picksAsText()), 'P(1st) disclosure missing from the export');
});

check('the header states the strategy and both scores', () => {
  const app = fixture(loadApp());
  const head = app.picksAsText().split('\n')[0];
  assert.ok(head.includes('2027'), 'season missing from header');
  assert.ok(head.includes('Maximise chance of winning'), 'strategy missing from header');
  // Whole points, not 9.9: the standard error on P(1st) is about 0.7pp, so a
  // decimal place implies a resolution fourteen times finer than the number
  // actually has. The fixture's 0.099 must print as 10%.
  assert.ok(/10% to finish first/.test(app.picksAsText()), 'P(1st) missing or over-precise');
  assert.ok(!/9\.9%/.test(app.picksAsText()), 'P(1st) printed finer than its own error');
});

check('no board means no export rather than a broken one', () => {
  const app = fixture(loadApp());
  app.state.rounds = null;
  assert.strictEqual(app.picksAsText(), '');
});


/* 2026-09 review: the export named a round's WINNERS under that round's own
 * label -- "Round of 64" headed the 32 teams who won their way OUT of it,
 * "Final Four" headed only the 2 who beat the other two, and no line said
 * which two teams had actually played. Fixed by printing both sides of every
 * game instead of shifting labels; ROUNDS[r] already named the right round. */
console.log('\ncopy output names both teams of each game (2026-09 review)');

/* Rounds 1-3 (Round of 32 through Elite 8) are left empty: this fixture only
 * needs to exercise the two kinds of round picksAsText() treats differently
 * -- region-pure (index 0, below REGION_SCOPED_ROUNDS) and cross-region
 * (index 4, the real position of the Final Four in a 6-round bracket) -- and
 * an empty games array for the rounds in between renders as an empty section
 * rather than throwing, so it costs nothing to leave them out. */
function regionFixture(app) {
  app.state.season = {
    teams: [
      { id: 'duke', name: 'Duke', seed: 1, region: 'East' },
      { id: 'akron', name: 'Akron', seed: 16, region: 'East' },
      { id: 'iowa', name: 'Iowa', seed: 8, region: 'West' },
      { id: 'utah', name: 'Utah', seed: 9, region: 'West' },
    ],
    strategies: [{ id: 'p1', label: 'Maximise chance of winning', p1: 0.099, ev: 874 }],
    p1_assumption: 'P(1st) assumes a 30-opponent pool with ESPN public pick behaviour.',
  };
  app.state.year = 2027;
  app.state.strategy = 'p1';
  app.state.rounds = [
    [{ a: 0, b: 1, win: 0 }, { a: 2, b: 3, win: 3 }],   // Round of 64: two region-pure games
    [], [], [],                                          // Round of 32, Sweet 16, Elite 8: unused
    [{ a: 0, b: 3, win: 0 }],                            // Final Four: crosses regions
  ];
  return app;
}

check('every game names both the winner and who it beat', () => {
  const app = regionFixture(loadApp());
  const text = app.picksAsText();
  assert.ok(text.includes('1 Duke over 16 Akron'), 'Round of 64 game 1 missing both sides:\n' + text);
  assert.ok(text.includes('9 Utah over 8 Iowa'), 'Round of 64 game 2 missing both sides:\n' + text);
  assert.ok(text.includes('1 Duke over 9 Utah'), 'Final Four missing both sides:\n' + text);
});

check('a region-scoped round groups its games under a region heading', () => {
  const app = regionFixture(loadApp());
  const text = app.picksAsText();
  const r64 = text.split('Round of 64\n')[1].split('\n\n')[0];
  assert.ok(/^\s*East\n\s*1 Duke over 16 Akron/m.test(r64), 'East heading must precede its game:\n' + r64);
  assert.ok(/West\n\s*9 Utah over 8 Iowa/.test(r64), 'West heading must precede its game:\n' + r64);
});

check('a round that crosses regions (Final Four) prints no region heading', () => {
  const app = regionFixture(loadApp());
  const text = app.picksAsText();
  const ff = text.split('Final Four\n')[1];
  // Neither team's own region ("East"/"West") should appear as a standalone
  // heading line ahead of the Final Four game -- that would misrepresent a
  // cross-region game as if it belonged to one region.
  assert.ok(!/^\s*(East|West)\s*$/m.test(ff), 'Final Four must not carry a region heading:\n' + ff);
  assert.ok(ff.includes('1 Duke over 9 Utah'));
});


/* ---------- addressable state ----------
 *
 * A static site's only sharing surface is its URL, and both bugs below shipped
 * in the first version of this: they restored something, so they looked like
 * they worked.
 */
console.log('\nurl state');

check('champ restores as a number, not a string', () => {
  // champ is a TEAM INDEX. Restoring "9" as a string fails every === against
  // the payload's 9, so the filter is dropped in silence -- the failure a
  // shared link is least likely to survive and least likely to report.
  const app = loadApp('#y=2026&o=p1&champ=9&pred=3');
  app.readHash();
  assert.strictEqual(app.state.pick.champ, 9, 'champ must be a number');
  assert.strictEqual(app.state.pick.pred, 3, 'pred must be a number');
});

check('src stays a string', () => {
  // The one filter whose values really are ids ("torvik", "elo").
  const app = loadApp('#y=2026&src=torvik');
  app.readHash();
  assert.strictEqual(app.state.pick.src, 'torvik');
});

check('a link with filters restores the FILTERED bracket', () => {
  // The second bug: filters restored, strategy left on 'ev', so the chips said
  // Florida while the board showed Michigan. A link that shows a different
  // bracket than it promised is worse than one that shows nothing.
  const app = loadApp('#y=2026&o=ev&champ=9');
  app.readHash();
  assert.strictEqual(app.state.objective, 'ev', 'objective must come from the link');
  assert.strictEqual(app.state.strategy, app.CUSTOM, 'filters present must mean CUSTOM');
});

check('a link with no filters keeps the plain objective', () => {
  const app = loadApp('#y=2026&o=ev');
  app.readHash();
  assert.strictEqual(app.state.strategy, 'ev');
});

check('the fitted model wins over filter restoration', () => {
  const app = loadApp('#y=2026&s=model&champ=9');
  app.readHash();
  assert.strictEqual(app.state.strategy, app.MODEL);
});

check('no hash is not an error', () => {
  const app = loadApp('');
  assert.strictEqual(app.readHash(), null);
});

check('the season comes back from the link', () => {
  const app = loadApp('#y=2013&o=p1');
  assert.strictEqual(app.readHash(), 2013);
});


/* ---------- which season opens ----------
 *
 * Previously asserted by grepping app.js for a regex, which proves the line
 * exists and nothing about what it does. This is the launch-day failure: the
 * site opening on last season's bracket on Selection Sunday 2027.
 */
console.log('\ndefault season');

const app0 = loadApp('');

check('opens on the newest READY season, not the newest listed', () => {
  // The 2027 shape exactly: listed, but not yet played.
  const seasons = [
    { year: 2025, status: 'ready' },
    { year: 2026, status: 'ready' },
    { year: 2027, status: 'not_started' },
  ];
  assert.strictEqual(app0.pickDefaultSeason(seasons), 2026);
});

check('opens on 2027 the moment it is built', () => {
  const seasons = [
    { year: 2026, status: 'ready' },
    { year: 2027, status: 'ready' },
  ];
  assert.strictEqual(app0.pickDefaultSeason(seasons), 2027);
});

check('a played-but-unavailable season is not chosen', () => {
  // 2012 has no picks archive; it must not be the landing page.
  const seasons = [
    { year: 2011, status: 'ready' },
    { year: 2012, status: 'unavailable' },
  ];
  assert.strictEqual(app0.pickDefaultSeason(seasons), 2011);
});

check('no ready season falls back to the newest listed, not to nothing', () => {
  const seasons = [{ year: 2027, status: 'not_started' }];
  assert.strictEqual(app0.pickDefaultSeason(seasons), 2027);
});

check('an empty index does not throw', () => {
  assert.strictEqual(app0.pickDefaultSeason([]), null);
  assert.strictEqual(app0.pickDefaultSeason(undefined), null);
});


/* ---------- how P(1st) is printed ---------- */
console.log('\nP(1st) display');

check('whole points, because the error is about 0.7pp', () => {
  const app = loadApp('');
  assert.strictEqual(app.p1Pct(0.099), '10%');
  assert.strictEqual(app.p1Pct(0.043), '4%');
});

check('an unlikely bracket is not printed as impossible', () => {
  // Rounding to whole points sent 0.4% to "0%", which reads as "cannot happen".
  // Three of 2026's candidates sit there.
  const app = loadApp('');
  assert.strictEqual(app.p1Pct(0.004), '<1%');
  assert.strictEqual(app.p1Pct(0.0049), '<1%');
  assert.strictEqual(app.p1Pct(0.005), '1%');
});

check('a genuine zero still prints as zero', () => {
  const app = loadApp('');
  assert.strictEqual(app.p1Pct(0), '0%');
});


/* refit() runs for EVERY season regardless of status -- setYear() calls it
 * before render() gets a chance to bail out on a season that has not
 * started -- and a `not_started` payload (docs/data/season_2027.json before
 * Selection Sunday) carries no `first_round` at all. refit() used to hand
 * that straight to bracketAdvancementProbs() as `undefined.length`, crashing
 * every load of a listed-but-not-yet-started season the instant the fitted
 * model had enough history to run (i.e. every season but the very first). */
console.log('\nrefit() on a not-started season (regression)');

check('refit() does not crash when the season has no bracket yet', () => {
  const app = loadApp();
  // Five rows is MIN_ROWS_PER_COL for one enabled column -- just enough for
  // fitLinear to report ok:true, so fitReady() is true and the crashing
  // branch is actually reached rather than short-circuited some other way.
  app.state.training = {
    keys: ['barthag'],
    years: [2020, 2021, 2022, 2023, 2024],
    games: [2020, 2021, 2022, 2023, 2024].map(y => ({ x: [1.2], m: 5, y })),
  };
  app.state.year = 2027;
  // The exact shape docs/data/season_2027.json has before Selection Sunday:
  // no `teams`, no `first_round`.
  app.state.season = { status: 'not_started', message: 'Not started', detail: '' };
  assert.doesNotThrow(() => app.refit());
  assert.strictEqual(app.state.advancement, null, 'no bracket means no advancement table, not a guess at one');
});

/* ---------- team drawer percentile (2026-09 site review) ----------
 *
 * The drawer used to place its marker at 50 + 16 z and show a raw number with
 * no direction: "Defense 91.0" gave no hint that lower is better, and the
 * marker was not a percentile of anything. Now it is a rank within the field
 * in the better direction, computed from the sign-corrected z column. */
console.log('\nteam drawer percentile');

check('best team is 100th, worst is 0th, in the BETTER direction', () => {
  const app = loadApp('');
  // z is already sign-corrected upstream: for a lower-is-better stat the
  // lowest raw carries the highest z. Raw is only consulted for missingness.
  const zs   = [ 2.0, -1.0, 0.5, -2.0];
  const raws = [90.0, 105.0, 98.0, 110.0];   // e.g. defensive efficiency
  assert.strictEqual(app.percentileInField(zs, raws, 0), 100);
  assert.strictEqual(app.percentileInField(zs, raws, 3), 0);
});

check('ties share a percentile rather than being ordered arbitrarily', () => {
  const app = loadApp('');
  const zs   = [1, 1, 0, -1];
  const raws = [1, 1, 2, 3];
  const a = app.percentileInField(zs, raws, 0);
  const b = app.percentileInField(zs, raws, 1);
  assert.strictEqual(a, b);
  assert.ok(a > app.percentileInField(zs, raws, 2));
});

check('a missing value is null, not ranked as average', () => {
  // A missing raw ships as z = 0. Ranking that would put a team with no data
  // at the median, which is what the old dot silently did.
  const app = loadApp('');
  const zs   = [1, 0, -1];
  const raws = [5, null, 3];
  assert.strictEqual(app.percentileInField(zs, raws, 1), null);
  // and it must not count toward the others' denominators
  assert.strictEqual(app.percentileInField(zs, raws, 0), 100);
  assert.strictEqual(app.percentileInField(zs, raws, 2), 0);
});

check('ordinal suffixes', () => {
  const app = loadApp('');
  assert.strictEqual(app.ordinal(1), '1st');
  assert.strictEqual(app.ordinal(2), '2nd');
  assert.strictEqual(app.ordinal(3), '3rd');
  assert.strictEqual(app.ordinal(11), '11th');
  assert.strictEqual(app.ordinal(12), '12th');
  assert.strictEqual(app.ordinal(13), '13th');
  assert.strictEqual(app.ordinal(21), '21st');
  assert.strictEqual(app.ordinal(100), '100th');
});

/* ---------- evaluated fitted bracket: shown only for THIS bracket ----------
 *
 * The payload may carry fitted_eval: the fitted bracket's P(1st)/EV from the
 * production referee (scripts/evaluate_fitted_bracket.py). Those numbers are
 * about one specific set of 63 picks. If the page's own fit produces any
 * other bracket -- a changed training matrix, a changed z column, a changed
 * tie -- the numbers must not appear. */
console.log('\nevaluated fitted bracket guard');

function fitted64(app) {
  // 64 teams; barthag z strictly decreasing with index, one training key with
  // a positive slope, so the fit sends the lower index through every game.
  const teams = [];
  for (let i = 0; i < 64; i++) teams.push({ id: 't' + i, name: 'T' + i, seed: (i % 16) + 1, region: 'R' + (i >> 4) });
  const z = { barthag: teams.map((_, i) => (32 - i) / 16) };
  app.state.training = {
    keys: ['barthag'],
    years: [2020, 2021, 2022, 2023, 2024],
    games: [2020, 2021, 2022, 2023, 2024].map(y => ({ x: [1.0], m: 6, y })),
  };
  app.state.year = 2027;
  app.state.strategy = app.MODEL;
  const chalk = [];
  let cur = teams.map((_, i) => i);
  for (let r = 0; r < 6; r++) { const n = []; for (let g = 0; g < cur.length; g += 2) n.push(cur[g]); chalk.push(n); cur = n; }
  app.state.season = {
    status: 'ready', teams, first_round: teams.map((_, i) => i), z, raw: {}, variables: [],
    fitted_eval: { kind: 'fitted_model_evaluated', w: chalk, ev: 900.1, p1: 0.0123 },
  };
  app.refit();
  return chalk;
}

check('an evaluation whose 63 picks match the live fit is returned', () => {
  const app = loadApp('');
  const chalk = fitted64(app);
  const live = app.solveByFit().map(games => games.map(g => g.win));
  // JSON, not deepStrictEqual: `live` was built inside the vm, whose Array
  // prototype is a different realm's, and strict deep equality checks that.
  assert.strictEqual(JSON.stringify(live), JSON.stringify(chalk), 'fixture must reproduce the evaluated bracket');
  const fe = app.fittedEval();
  assert.ok(fe && !fe.stale, 'expected the evaluation');
  assert.strictEqual(fe.p1, 0.0123);
});

check('one different pick anywhere makes it stale, not slightly wrong', () => {
  const app = loadApp('');
  fitted64(app);
  app.state.season.fitted_eval.w[3][2] += 1;      // an Elite 8 winner, swapped
  const fe = app.fittedEval();
  assert.ok(fe && fe.stale === true, 'expected stale');
});

check('a wrong kind is ignored outright', () => {
  const app = loadApp('');
  fitted64(app);
  app.state.season.fitted_eval.kind = 'candidate';
  assert.strictEqual(app.fittedEval(), null);
});

check('no evaluation on the payload is simply null', () => {
  const app = loadApp('');
  fitted64(app);
  delete app.state.season.fitted_eval;
  assert.strictEqual(app.fittedEval(), null);
});

console.log('strict picks expansion (audit 2026-09, Step 4, F4-7)');

/* The 64-team walk in solveFromPicks: the picks list must name exactly one
 * team of every game. The old `has(a) ? a : b` silently produced a bracket
 * from any list at all. */
function season64(app) {
  const teams = [];
  for (let i = 0; i < 64; i++) teams.push({ id: 't' + i, name: 'T' + i, seed: (i % 16) + 1 });
  const first_round = teams.map((_, i) => i);
  // chalk picks: index 2g wins every game
  const picks = [];
  let cur = first_round.slice();
  for (let r = 0; r < 6; r++) {
    const next = [];
    for (let g = 0; g < cur.length; g += 2) next.push(cur[g]);
    picks.push(next);
    cur = next;
  }
  app.state.season = { teams, first_round, strategies: [], pool_optimized: picks };
  app.state.strategy = 'p1';
  return { picks };
}

check('a well-formed picks list expands to the bracket it describes', () => {
  const app = loadApp();
  season64(app);
  const rounds = app.solveFromPicks();
  assert.strictEqual(rounds.length, 6);
  assert.strictEqual(rounds[5][0].win, 0);
});

check('a picks list naming neither team of a game throws instead of guessing', () => {
  const app = loadApp();
  const { picks } = season64(app);
  picks[0][0] = 63;               // game 0 is t0 vs t1; neither is picked now
  assert.throws(() => app.solveFromPicks(), /neither picked/);
});

check('a picks list naming both teams of a game throws instead of guessing', () => {
  const app = loadApp();
  const { picks } = season64(app);
  picks[0].push(1);               // game 0 now has both t0 and t1 as R64 winners
  assert.throws(() => app.solveFromPicks(), /both picked/);
});


/* ---------- model sensitivity copy (Phase B, preregistered) ----------
 * The preregistration forbids causal or importance wording for exclusion
 * refits. The renderer's user-facing strings sit between two markers in
 * app.js; this scans them. */
console.log('\nmodel sensitivity wording');

check('no causal or importance words in the sensitivity panel', () => {
  const src = fs.readFileSync(path.join(__dirname, '..', 'docs', 'app.js'), 'utf8');
  const a = src.indexOf('/* SENSITIVITY-COPY-START'), b = src.indexOf('/* SENSITIVITY-COPY-END');
  assert.ok(a > 0 && b > a, 'markers missing');
  const block = src.slice(a, b)
    .replace(/\/\*[\s\S]*?\*\//g, '')        // block comments
    .replace(/^\s*\/\/.*$/gm, '');           // line comments
  for (const bad of ['contribut', 'causal', 'cause of', 'importance', 'important variable', 'explains', 'accounts for', 'percentage points from', 'effect of']) {
    assert.ok(!block.toLowerCase().includes(bad), `forbidden wording "${bad}" in the sensitivity panel`);
  }
  assert.ok(block.includes('refit excluding'), 'the required phrasing is missing');
});

console.log(`\n${passed} checks passed`);
