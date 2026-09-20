/* Checks what is left of docs/app.js after the 2026-09-19 redesign (see
 * docs/index2.html): the year nav, the rule search, the adjust filters, the
 * team drawer and the methodology panel were removed in favour of a card
 * picker, a real leaderboard and a decorative bracket. This file used to also
 * cover the pick exporter, the URL-restorable filters, the team drawer
 * percentile, the model-sensitivity panel and the whole rule-search
 * subsystem -- all of that surface is gone from app.js, so those checks went
 * with it rather than being left to assert on functions that no longer
 * exist.
 *
 * What is still real and still checked here: which season opens by default
 * (pickDefaultSeason), the P(1st) display rounding (p1Pct), refit() not
 * crashing on a not-started season, the fitted-bracket evaluation guard
 * (fittedEval), and ordinal suffixes.
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

function loadApp() {
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
    module: undefined,  // fit.js only attaches to module.exports if this exists
  };
  vm.createContext(ctx);
  vm.runInContext(fitSrc, ctx);
  vm.runInContext(src, ctx);
  // Top-level `const` lives in the script's lexical scope, not on the context
  // object, so reach it by evaluating in that same scope.
  vm.runInContext(
    'globalThis.__api = { state, MODEL, pickDefaultSeason, p1Pct, refit, ordinal, fittedEval, solveByFit, winProb, strategyRows, fieldPending };', ctx);
  return ctx.__api;
}

/* ---------- which season opens ----------
 *
 * Previously asserted by grepping app.js for a regex, which proves the line
 * exists and nothing about what it does. This is the launch-day failure: the
 * site opening on last season's bracket on Selection Sunday 2027.
 */
console.log('\ndefault season');

const app0 = loadApp('');

check('opens on the newest listed season once it has a bracket', () => {
  const seasons = [
    { year: 2025, status: 'ready' },
    { year: 2026, status: 'ready' },
    { year: 2027, status: 'not_started' },
  ];
  assert.strictEqual(app0.pickDefaultSeason(seasons), 2027, 'listed and pending: still the front door (fieldPending())');
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

/* ---------- strategy cards under a pending field ----------
 *
 * No year nav any more (see docs/index2.html, 2026-09-19): the page always
 * shows pickDefaultSeason()'s season, and a listed-but-not-started one with
 * played seasons before it is "pending" -- the 3 cards are listed for what
 * they will be rather than the page going empty. */
console.log('\nstrategy cards under a pending field');

check('a listed season with played seasons before it is pending, not empty; the cards list what will be', () => {
  const app = loadApp('');
  app.state.year = 2027;
  app.state.season = { year: 2027, status: 'not_started', message: 'The 2027 season hasn\'t started yet.' };
  app.state.seasonsIndex = [
    { year: 2025, status: 'ready' }, { year: 2026, status: 'ready' }, { year: 2027, status: 'not_started' },
  ];
  assert.strictEqual(app.fieldPending(), true);
  assert.strictEqual(app.fieldPending({ year: 2022, status: 'unavailable' }), false);
  app.state.seasonsIndex = [{ year: 2027, status: 'not_started' }];
  assert.strictEqual(app.fieldPending(), false, 'nothing played before it: nothing pending, just unavailable');
  app.state.seasonsIndex = [
    { year: 2025, status: 'ready' }, { year: 2026, status: 'ready' }, { year: 2027, status: 'not_started' },
  ];
  const rows = app.strategyRows();
  assert.strictEqual(rows.length, 4, 'the three validated strategies plus the experimental rule search');
  assert.strictEqual(JSON.stringify(rows.map(r => r.id)), JSON.stringify(['p1', 'ev', 'model', 'rule']));
  assert.ok(rows.every(r => r.pending === true));
  assert.ok(rows.every(r => r.p1 === null && r.ev === null && r.champion === null));
});

console.log(`\n${passed} checks passed`);
