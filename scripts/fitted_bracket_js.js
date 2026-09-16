#!/usr/bin/env node
/* Produce the browser's "fitted model" bracket by RUNNING THE BROWSER'S CODE.
 *
 * scripts/evaluate_fitted_bracket.py needs the exact bracket a visitor sees
 * under the Fitted strategy, so it can be scored by the same Python referee
 * as the other two cards. There are two ways to get it: port fit.js's ridge
 * fit, walk-forward window, calibration and game-by-game tie rule to Python,
 * or execute fit.js + app.js themselves. The port already exists in part
 * (src/prediction/pit_production_model.py) and its parity test covers
 * aggregate walk-forward metrics -- not per-game picks, not the p == 0.5 tie
 * rule, not app.js's silent handling of keys missing from the matrix. Any of
 * those could produce a bracket that differs from the page by one game, and
 * a P(1st) for a bracket that is not the one on screen is worse than no
 * number. So: the same two files the <script> tags load, evaluated in a vm
 * with the same stubs tests/test_picks_export.js uses, with the same inputs
 * the browser fetches (docs/data/training.json, docs/data/season_YYYY.json).
 *
 * Usage: node scripts/fitted_bracket_js.js <year> [docsDir]
 * Prints one JSON object to stdout:
 *   { year, w: [[team index, ...] x 6 rounds], keys, beta, sigma, n, calibration, oos }
 * w[r] is the list of winners of round r, as indices into season.teams --
 * the same encoding the candidate artifact's `w` uses.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const year = parseInt(process.argv[2], 10);
if (!Number.isFinite(year)) {
  console.error('usage: node scripts/fitted_bracket_js.js <year> [docsDir]');
  process.exit(2);
}
const docs = path.resolve(process.argv[3] || path.join(__dirname, '..', 'docs'));

const fitSrc = fs.readFileSync(path.join(docs, 'fit.js'), 'utf8');
const appSrc = fs.readFileSync(path.join(docs, 'app.js'), 'utf8');
const training = JSON.parse(fs.readFileSync(path.join(docs, 'data', 'training.json'), 'utf8'));
const season = JSON.parse(fs.readFileSync(path.join(docs, 'data', `season_${year}.json`), 'utf8'));
if (season.status !== 'ready') {
  console.error(`season ${year} is ${season.status}; nothing to fit`);
  process.exit(3);
}

const noop = () => {};
const ctx = {
  console: { log: noop, warn: noop, error: noop },
  setTimeout, clearTimeout,
  fetch: () => new Promise(() => {}),          // init() awaits forever; no DOM is touched
  document: { getElementById: () => null, querySelector: () => null, querySelectorAll: () => [], addEventListener: noop },
  window: { isSecureContext: false, matchMedia: () => ({ matches: false }) },
  navigator: {},
  URLSearchParams,
  location: { hash: '' },
  history: { replaceState: noop },
  module: undefined,
};
vm.createContext(ctx);
vm.runInContext(fitSrc, ctx);
vm.runInContext(appSrc, ctx);
vm.runInContext('globalThis.__api = { state, refit, solveByFit, MODEL, CANONICAL_KEYS };', ctx);
const api = ctx.__api;

api.state.training = training;
api.state.season = season;
api.state.year = year;
api.state.strategy = api.MODEL;
api.refit();
const f = api.state.fit;
if (!f || !f.ok) {
  console.error(`fit not ok for ${year}: ${f && f.reason}`);
  process.exit(4);
}
const rounds = api.solveByFit();

process.stdout.write(JSON.stringify({
  year,
  w: rounds.map(games => games.map(g => g.win)),
  keys: f.keys,
  dropped: f.dropped,
  beta: f.beta,
  sigma: f.sigma,
  n: f.n,
  calibration: f.oos ? { a: f.oos.calibration.a, nu: f.oos.calibration.nu } : null,
  oos: f.oos ? { accuracy: f.oos.accuracy, n: f.oos.n, seasons: f.oos.seasons } : null,
}) + '\n');
