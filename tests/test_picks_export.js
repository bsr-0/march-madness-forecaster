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

function loadApp(hash, opts = {}) {
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
    // init() awaits this forever, so it never reaches the DOM. Tests that
    // drive the rule search pass a fetch that serves season payloads instead.
    fetch: opts.fetch || (() => new Promise(() => {})),
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
  // The rule-search setters end by redrawing the strategy cards and the
  // board, which need a DOM; tests of the search state replace those two
  // renderers (function declarations, so reassignable in the script scope)
  // and check state.rule directly.
  if (opts.noRender) vm.runInContext('render = () => {}; renderStrategies = () => {};', ctx);
  // Top-level `const` lives in the script's lexical scope, not on the context
  // object, so reach it by evaluating in that same scope.
  vm.runInContext(
    'globalThis.__api = { state, picksAsText, ROUNDS, readHash, writeHash, CUSTOM, MODEL, solveFromPicks, '
    + 'pickDefaultSeason, p1Pct, refit, percentileInField, ordinal, fittedEval, solveByFit, solveBracket, sensitivity, winProb, RULE, ruleStrategy, strategyRows, currentStrategy, '
    + 'ensureRuleSearch, ruleRange, ruleKeys, ruleRoundLabel, setRuleCheckpoint, setRuleMode, setRuleHand, setRuleRange, setRuleLast, setRuleN, setRuleRank, setRuleKey, setRuleKeysAll };', ctx);
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

check('no causal or importance words anywhere in the Fitted Model surface', () => {
  // Terminology audit (2026-09-16): the equation, reliability table,
  // Explore panel and sensitivity panel describe observed relationships,
  // refit differences and pick sensitivity -- never contribution, importance,
  // impact, cause, drive, improve, explain, optimal, predictive power,
  // overrule, or effect. Scans the user-facing renderers with comments
  // stripped; `value` is allowed ("pre-tournament value of this variable").
  const src = fs.readFileSync(path.join(__dirname, '..', 'docs', 'app.js'), 'utf8');
  const body = name => {
    const i = src.indexOf('function ' + name + '(');
    assert.ok(i >= 0, name + ' missing');
    const rest = src.slice(i + 9);
    const end = rest.search(/\n(function |\/\* ----------|const [A-Z_]+ = )/);
    return end < 0 ? rest : rest.slice(0, end);
  };
  const block = ['equationHTML', 'reliabilityHTML', 'renderExplore', 'sensitivityHTML', 'advancementHTML', 'probTitle', 'renderRulePanel', 'ruleStrategy']
    .map(body).join('\n')
    .replace(/\/\*[\s\S]*?\*\//g, '')        // block comments
    .replace(/^\s*\/\/.*$/gm, '')            // line comments
    .replace(/causalWalkForward/g, '');      // a function name, not copy
  for (const bad of ['contribut', 'importan', 'impact', 'causal', ' cause', 'drives', 'improve', 'explain', 'optimal', 'predictive power', 'overrul', 'effect', 'better variable', 'accounts for']) {
    assert.ok(!block.toLowerCase().includes(bad), `forbidden wording "${bad}" in the Fitted Model surface`);
  }
  assert.ok(block.includes('refit excluding'), 'the required phrasing is missing');
});

/* ---------- "picks changed" is a deterministic comparison of two solves ---------- */
console.log('\npicks changed: two independent deterministic solves');

// 64 teams, two training keys. `dup` is a copy of `barthag` in both the
// training rows and the field, so excluding either must leave the walk
// unchanged; `other` (third fixture) disagrees with barthag for some teams.
function sensFixture(app, keys, zFor, mode) {
  const teams = [];
  for (let i = 0; i < 64; i++) teams.push({ id: 't' + i, name: 'T' + i, seed: (i % 16) + 1, region: 'R' + (i >> 4) });
  const years = [];
  for (let y = 2010; y <= 2024; y++) years.push(y);
  let seed = 11;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  const games = [];
  for (const y of years) for (let g = 0; g < 63; g++) {
    const v = rnd() * 4, w = rnd() * 4;
    // mode 'dup': the second key is an exact copy of the first in training
    // AND in the field (zFor decides the field). mode 'indep': the second key
    // is an independent signal the margin also depends on.
    const x = keys.map((k, j) => (j === 0 || mode === 'dup' ? v : w));
    games.push({ y, x, m: Math.round(3 * v + (mode === 'indep' ? 2 * w : 0) + rnd() * 6) || 1, r: 'R64' });
  }
  app.state.training = { keys, years, games };
  app.state.year = 2025;
  app.state.strategy = app.MODEL;
  const z = {};
  for (const k of keys) z[k] = teams.map((_, i) => zFor(k, i));
  app.state.season = { status: 'ready', teams, first_round: teams.map((_, i) => i), z, raw: {}, variables: keys.map(k => ({ key: k, label: k, group: 'g', higher_better: true })) };
  app.refit();
  return app;
}

check('the count equals a slot-by-slot diff of two solveBracket() walks', () => {
  const app = loadApp('');
  // Two canonical keys (refit() only fits CANONICAL_KEYS): t_rank is an
  // independent signal in training and disagrees with barthag for some teams
  // in the field, so the exclusion refits genuinely change picks.
  sensFixture(app, ['barthag', 't_rank'], (k, i) => (k === 'barthag' ? (32 - i) / 16 : ((i % 7 === 0 ? -1 : 1) * (32 - i)) / 16), 'indep');
  const sens = app.sensitivity();
  for (const key of ['barthag', 't_rank']) {
    const e = sens.byKey[key];
    const base = app.solveBracket(app.winProb);
    const alt = app.solveBracket(e.pFn);
    let n = 0;
    base.forEach((games, r) => games.forEach((g, i) => { if (alt[r][i].win !== g.win) n++; }));
    assert.strictEqual(e.changed, n, key);
  }
});

check('excluding an exact duplicate column changes zero picks', () => {
  const app = loadApp('');
  // t_rank made identical to barthag everywhere (training and field).
  sensFixture(app, ['barthag', 't_rank'], (k, i) => (32 - i) / 16, 'dup');
  const sens = app.sensitivity();
  assert.strictEqual(sens.byKey.barthag.changed, 0);
  assert.strictEqual(sens.byKey.t_rank.changed, 0);
});

check('the comparison is deterministic: recomputing gives identical counts and brackets', () => {
  const app = loadApp('');
  sensFixture(app, ['barthag', 't_rank'], (k, i) => (k === 'barthag' ? (32 - i) / 16 : ((i % 5 === 0 ? -1 : 1) * (32 - i)) / 16), 'indep');
  const a = app.sensitivity();
  app.state.sens = null;
  const b = app.sensitivity();
  for (const key of ['barthag', 't_rank']) assert.strictEqual(a.byKey[key].changed, b.byKey[key].changed, key);
  assert.strictEqual(JSON.stringify(a.base.map(r => r.map(g => g.win))), JSON.stringify(b.base.map(r => r.map(g => g.win))));
});

check('the baseline bracket is solved inside sensitivity(), not read from state.rounds', () => {
  const app = loadApp('');
  sensFixture(app, ['barthag', 't_rank'], (k, i) => (32 - i) / 16, 'dup');
  app.state.rounds = null;                      // nothing rendered
  const sens = app.sensitivity();
  assert.ok(Array.isArray(sens.base) && sens.base.length === 6);
  assert.strictEqual(sens.byKey.barthag.changed, 0);
});

/* ---------- rule search: an experimental strategy, fenced ---------- */
console.log('\nrule search strategy fencing');

check('s=rule round-trips through the URL and is not turned into a filter state', () => {
  const app = loadApp('#y=2026&s=rule&champ=9');
  app.readHash();
  assert.strictEqual(app.state.strategy, app.RULE);
  app.state.season = { teams: [], strategies: [], first_round: [] };
  app.writeHash();
  assert.ok(ctxHash.value.includes('s=rule'), ctxHash.value);
});

check('the rule strategy carries no p1, no ev, and no record', () => {
  const app = loadApp('');
  fitted64(app);                                   // a season with 64 teams and a fit
  app.state.strategy = app.RULE;
  app.state.rule.result = {
    key: 'k', mode: 'search', usedSeasons: [2025], requested: [2023, 2024, 2025], backedOff: true, nRules: 1, lastRound: 3, scored: 1, nOutside: 4,
    brackets: [{ seq: ['barthag'], rounds: app.solveBracket(app.winProb), picks: app.solveBracket(app.winProb).map(g => g.map(x => x.win)), complexity: [1, 0], outside: { k: 0, m: 4, years: [] } }],
  };
  const st = app.currentStrategy();
  assert.strictEqual(st.id, app.RULE);
  assert.strictEqual(st.p1, undefined);
  assert.strictEqual(st.ev, undefined);
  assert.strictEqual(st.picks.length, 6);
  const row = app.strategyRows().find(r => r.id === app.RULE);
  assert.strictEqual(row.p1, null); assert.strictEqual(row.ev, null); assert.strictEqual(row.record, null);
  assert.ok(/not scored/i.test(row.kind));
  // and the export says so on its second line
  app.state.rounds = st.picks.map((w, r) => w.map((win, i) => ({ a: win, b: win, win })));
  const text = app.picksAsText();
  assert.ok(/not scored against the pool/i.test(text.split('\n')[1]), text.split('\n')[1]);
});

check('with no result yet, the rule strategy resolves to nothing rather than to another bracket', () => {
  const app = loadApp('');
  fitted64(app);
  app.state.strategy = app.RULE; app.state.rule.result = null;
  assert.strictEqual(app.ruleStrategy(), null);
  assert.strictEqual(app.currentStrategy(), null);
});

/* ---------- rule search: the panel's controls, driven end to end ---------- */
console.log('\nrule search controls');

/* Synthetic seasons for the search: 64 teams, two criteria. `a` falls with
 * team index, so under it the lower index wins every game (champion 0);
 * `b` is its mirror (champion 63). A season's `actual` is one of those two
 * brackets, so the only rule that reproduces the Final Four onward is the
 * matching criterion in every constrained round. */
function ruleSeasonPayload(champCrit) {
  const teams = [];
  for (let i = 0; i < 64; i++) teams.push({ id: 't' + i, name: 'T' + i, seed: (i % 16) + 1, region: 'R' + (i >> 4) });
  const a = teams.map((_, i) => (32 - i) / 16), b = a.slice().reverse();
  const actual = [];
  let cur = teams.map((_, i) => i);
  for (let r = 0; r < 6; r++) { const n = []; for (let g = 0; g < cur.length; g += 2) n.push(champCrit === 'a' ? cur[g] : cur[g + 1]); actual.push(n); cur = n; }
  return { status: 'ready', teams, first_round: teams.map((_, i) => i), z: { a, b }, raw: {}, actual,
           variables: [{ key: 'a', label: 'A', group: 'G' }, { key: 'b', label: 'B', group: 'G' }], strategies: [] };
}
const RULE_SEASONS = { 2023: 'b', 2024: 'a', 2025: 'a', 2026: 'a' };

/* A fetch serving those payloads; `delayFor(callNo)` lets a test make the
 * first request the slowest. */
function ruleFetch(delayFor = () => 0) {
  let n = 0;
  return url => new Promise(resolve => {
    const y = Number((url.match(/season_(\d+)\.json/) || [])[1]);
    if (!y) return;                                    // init()'s seasons.json: keep it pending, as loadApp does
    const d = delayFor(n++);
    setTimeout(() => resolve({ ok: true, json: async () => ruleSeasonPayload(RULE_SEASONS[y]) }), d);
  });
}

function ruleApp(delayFor) {
  const app = loadApp('', { fetch: ruleFetch(delayFor), noRender: true });
  app.state.seasonsIndex = [2022, 2023, 2024, 2025, 2026, 2027].map(y => ({ year: y, status: y === 2022 ? 'unavailable' : y === 2027 ? 'not_started' : 'ready' }));
  app.state.year = 2026;
  app.state.season = ruleSeasonPayload('a');
  app.state.strategy = app.RULE;
  return app;
}

check('the fit range is clamped to played seasons before the displayed one and defaults to the last three', () => {
  const app = ruleApp();
  let r = app.ruleRange();
  assert.deepStrictEqual([r.from, r.to, [...r.fit]], [2023, 2025, [2023, 2024, 2025]]);
  app.state.rule.from = 2024; app.state.rule.to = 2026;              // 2026 is the displayed season: not selectable
  r = app.ruleRange();
  assert.deepStrictEqual([r.from, r.to, [...r.fit]], [2024, 2025, [2024, 2025]]);
  app.state.rule.from = 2025; app.state.rule.to = 2023;              // reversed pickers still make a range
  r = app.ruleRange();
  assert.deepStrictEqual([r.from, r.to], [2023, 2025]);
  app.state.year = 2023;                                             // nothing played before it
  r = app.ruleRange();
  assert.deepStrictEqual([r.from, r.to, [...r.fit], [...r.played]], [null, null, [], []]);
});

check('at least one of Elite Eight / Final Four stays a checkpoint', () => {
  const app = ruleApp();
  app.state.rule.checkpoints = [3, 4, 5];
  app.setRuleCheckpoint(3, false);                                   // would leave finalists + champion only
  assert.deepStrictEqual([...app.state.rule.checkpoints], [3, 4, 5]);
  app.setRuleCheckpoint(2, true); app.setRuleCheckpoint(3, false);
  assert.deepStrictEqual([...app.state.rule.checkpoints], [2, 4, 5]);
  app.setRuleCheckpoint(5, false);
  assert.deepStrictEqual([...app.state.rule.checkpoints], [2, 4]);
});

check('eligible criteria: a set never empties, "all" clears the restriction', () => {
  const app = ruleApp();
  assert.deepStrictEqual([...app.ruleKeys()], ['a', 'b', 'seed']);
  app.setRuleKey('a', false);
  assert.deepStrictEqual([...app.ruleKeys()], ['b', 'seed']);
  app.setRuleKeysAll(false);
  assert.deepStrictEqual([...app.ruleKeys()], ['seed']);
  app.setRuleKey('seed', false);                                     // refused: nothing left to search over
  assert.deepStrictEqual([...app.ruleKeys()], ['seed']);
  app.setRuleKeysAll(true);
  assert.strictEqual(app.state.rule.keys, null);
});

const asyncChecks = [];
function checkAsync(name, fn) { asyncChecks.push([name, fn]); }

checkAsync('the search fits the chosen range, backs off, and counts seasons outside it', async () => {
  const app = ruleApp();
  await app.ensureRuleSearch();
  let r = app.state.rule.result;
  assert.strictEqual(r.mode, 'search');
  assert.deepStrictEqual([...r.requested], [2023, 2024, 2025]);
  assert.deepStrictEqual([...r.usedSeasons], [2024, 2025], '2023 is the mirror: dropped by the back-off');
  assert.strictEqual(r.backedOff, true);
  // Seeds in the fixture pick the lower index too, so `a` and `seed` are
  // interchangeable: 2^6 rules, all giving the one bracket, of which the
  // all-`a` one sorts first.
  assert.strictEqual(r.nRules, 64);
  assert.strictEqual(r.scored, 1); assert.strictEqual(r.brackets.length, 1);
  assert.deepStrictEqual([...r.brackets[0].seq], ['a', 'a', 'a', 'a', 'a', 'a']);
  assert.strictEqual(JSON.stringify(r.brackets[0].outside), JSON.stringify({ k: 0, m: 1, years: [] }), 'only 2023 lies outside, and it does not fit');
  assert.strictEqual(r.brackets[0].picks[5][0], 0);
  const st = app.ruleStrategy();
  assert.strictEqual(JSON.stringify(st.prior), JSON.stringify({ k: 0, m: 1 }));
  assert.ok(/2024–2025/.test(st.note) && /2023–2025 has none/.test(st.note), st.note);
  assert.strictEqual(app.ruleRoundLabel(0, ' · '), ' · A');
  // A range with the mirror season alone: the rule is `b`, and both other seasons lie outside it.
  app.setRuleRange(2023, 2023); await app.ensureRuleSearch();
  r = app.state.rule.result;
  assert.strictEqual(r.backedOff, false);
  assert.strictEqual(r.nRules, 1);
  assert.deepStrictEqual([...r.brackets[0].seq], ['b', 'b', 'b', 'b', 'b', 'b']);
  assert.strictEqual(JSON.stringify(r.brackets[0].outside), JSON.stringify({ k: 0, m: 2, years: [] }));
  assert.strictEqual(r.brackets[0].picks[5][0], 63);
  // Restricting the criteria to one that fits no season ending the range leaves no bracket, and the strategy resolves to nothing.
  app.setRuleLast(3); app.setRuleKeysAll(false); app.setRuleKey('b', true); app.setRuleKey('seed', false);
  assert.deepStrictEqual([...app.ruleKeys()], ['b']);
  await app.ensureRuleSearch();
  r = app.state.rule.result;
  assert.strictEqual(r.brackets.length, 0);
  assert.strictEqual(r.nRules, 0);
  assert.strictEqual(app.ruleStrategy(), null);
});

checkAsync('composing by hand applies the rule and reports every played season it reproduces', async () => {
  const app = ruleApp();
  app.setRuleMode('hand');
  assert.deepStrictEqual([...app.state.rule.hand], ['a', 'a', 'a', 'a', 'a', 'a'], 'starts from the season\'s first listed variable, not a hardcoded key');
  for (let i = 0; i < 6; i++) app.setRuleHand(i, 'a');
  await app.ensureRuleSearch();
  const r = app.state.rule.result;
  assert.strictEqual(r.mode, 'hand');
  assert.strictEqual(r.brackets.length, 1);
  assert.strictEqual(JSON.stringify(r.brackets[0].per.map(x => [x.year, x.ok])), JSON.stringify([[2023, false], [2024, true], [2025, true]]));
  const st = app.ruleStrategy();
  assert.strictEqual(JSON.stringify(st.prior), JSON.stringify({ k: 2, m: 3 }));
  assert.ok(/composed by hand/.test(st.note) && /2 of 3 played seasons before 2026 \(2024, 2025\)/.test(st.note), st.note);
  assert.strictEqual(st.p1, undefined); assert.strictEqual(st.ev, undefined);
  app.setRuleHand(5, 'b'); await app.ensureRuleSearch();
  assert.deepStrictEqual([...app.state.rule.result.brackets[0].per.map(x => x.ok)], [false, false, false], 'a b final flips the champion everywhere');
  assert.strictEqual(app.ruleRoundLabel(5, ' · '), ' · B');
});

checkAsync('an overlapping earlier call abandons; the result matches the latest controls', async () => {
  // The first fetch is the slowest, so the first ensureRuleSearch() resumes
  // after the second has finished. Without the token it would then run the
  // search and overwrite the by-hand result with one for stale inputs.
  const app = ruleApp(n => (n === 0 ? 30 : 0));
  const first = app.ensureRuleSearch();
  app.setRuleMode('hand');
  const second = app.ensureRuleSearch();
  await Promise.all([first, second]);
  assert.strictEqual(app.state.rule.result.mode, 'hand');
  assert.strictEqual(app.state.rule.busy, false);
});

(async () => {
  for (const [name, fn] of asyncChecks) {
    try { await fn(); passed++; console.log('  ok   ' + name); }
    catch (e) { console.error('  FAIL ' + name + '\n       ' + e.message); process.exitCode = 1; }
  }
  console.log(`\n${passed} checks passed`);
})();
