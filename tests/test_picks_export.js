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

function loadApp() {
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
  };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  // Top-level `const` lives in the script's lexical scope, not on the context
  // object, so reach it by evaluating in that same scope.
  vm.runInContext('globalThis.__api = { state, picksAsText, ROUNDS };', ctx);
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
  assert.ok(/9\.9% to finish first/.test(app.picksAsText()), 'P(1st) missing');
});

check('no board means no export rather than a broken one', () => {
  const app = fixture(loadApp());
  app.state.rounds = null;
  assert.strictEqual(app.picksAsText(), '');
});

console.log(`\n${passed} checks passed`);
