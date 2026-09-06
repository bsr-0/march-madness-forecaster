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
  };
  if (hash) ctx.location.hash = hash;
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  // Top-level `const` lives in the script's lexical scope, not on the context
  // object, so reach it by evaluating in that same scope.
  vm.runInContext(
    'globalThis.__api = { state, picksAsText, ROUNDS, readHash, writeHash, CUSTOM, MODEL, '
    + 'pickDefaultSeason, p1Pct };', ctx);
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

console.log(`\n${passed} checks passed`);
