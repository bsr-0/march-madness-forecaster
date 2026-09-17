/* Checks for pairwiseCorrelations() in docs/fit.js.
 *
 * This is the check stability() cannot do: two coefficients can each be
 * perfectly sign-stable across every held-out fold and still not mean what
 * they look like, because the two COLUMNS carry almost the same information
 * (Overall rating and National rank correlate at ~0.99 on real data) and the
 * fit is free to draw any split between them that sums to the right net
 * effect. Wrong here means the equation's "unstable" marking silently misses
 * exactly the two variables the 2026-09 site review flagged as misleading.
 *
 * Run: node tests/test_collinearity.js
 */

const assert = require('assert');
const path = require('path');
const F = require(path.join(__dirname, '..', 'docs', 'fit.js'));

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
const close = (a, b, tol, what) =>
  assert.ok(Math.abs(a - b) < tol, `${what || ''} expected ${b}, got ${a} (tol ${tol})`);

console.log('pairwise correlations');

check('two identical columns correlate at exactly 1', () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({ y: 2000, x: [i, i] }));
  const corr = F.pairwiseCorrelations(rows, [0, 1], null);
  close(corr[0][1], 1, 1e-9);
  close(corr[1][0], 1, 1e-9);
  close(corr[0][0], 1, 1e-9);
  close(corr[1][1], 1, 1e-9);
});

check('two exactly opposite columns correlate at exactly -1', () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({ y: 2000, x: [i, -i] }));
  const corr = F.pairwiseCorrelations(rows, [0, 1], null);
  close(corr[0][1], -1, 1e-9);
});

check('a column with no variance correlates at 0, not NaN', () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({ y: 2000, x: [i, 7] }));
  const corr = F.pairwiseCorrelations(rows, [0, 1], null);
  assert.strictEqual(corr[0][1], 0);
  assert.ok(!Number.isNaN(corr[0][1]));
});

check('two independent alternating columns correlate near 0', () => {
  // Deterministic, not random, so the test cannot flake: a 4-cycle in x0 and
  // an unrelated 3-cycle in x1 share no linear relationship over their LCM.
  const rows = Array.from({ length: 120 }, (_, i) => ({
    y: 2000, x: [i % 4, i % 3],
  }));
  const corr = F.pairwiseCorrelations(rows, [0, 1], null);
  assert.ok(Math.abs(corr[0][1]) < 0.05, `expected near 0, got ${corr[0][1]}`);
});

check('matches a hand-computed correlation on real-shaped data', () => {
  const a = [2, 4, 6, 8, 10];
  const b = [1, 2, 4, 4, 5];   // strongly but not perfectly related to a
  const rows = a.map((v, i) => ({ y: 2000, x: [v, b[i]] }));
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < a.length; i++) {
    cov += (a[i] - ma) * (b[i] - mb);
    va += (a[i] - ma) ** 2;
    vb += (b[i] - mb) ** 2;
  }
  const expected = cov / Math.sqrt(va * vb);
  const corr = F.pairwiseCorrelations(rows, [0, 1], null);
  close(corr[0][1], expected, 1e-9);
});

check('asOf restricts to strictly earlier rows, same as fitLinear', () => {
  // Rows from 2026 onward carry a relationship the training window (< 2020)
  // must never see; if asOf leaked, corr would be pulled toward 1.
  const early = Array.from({ length: 20 }, (_, i) => ({ y: 2010 + (i % 5), x: [i % 4, i % 3] }));
  const late = Array.from({ length: 20 }, (_, i) => ({ y: 2026, x: [i, i] }));
  const corr = F.pairwiseCorrelations([...early, ...late], [0, 1], 2020);
  assert.ok(Math.abs(corr[0][1]) < 0.1, `late rows leaked into the correlation: ${corr[0][1]}`);
});

check('no training rows yields null rather than a divide-by-zero', () => {
  const corr = F.pairwiseCorrelations([{ y: 2026, x: [1, 2] }], [0, 1], 2000);
  assert.strictEqual(corr, null);
});


console.log('\nvariable record (what one variable predicts, walk-forward)');

check('better-value team win rate, overall and by round, with SE', () => {
  const rows = [
    { y: 2010, x: [1.0], m: 5, r: 'R64' },    // better team won
    { y: 2010, x: [-2.0], m: -3, r: 'R64' },  // better team (team2) won
    { y: 2010, x: [0.5], m: -1, r: 'R32' },   // better team lost
    { y: 2010, x: [0.0], m: 4, r: 'R32' },    // no edge: excluded
    { y: 2026, x: [3.0], m: 20, r: 'NCG' },   // not before asOf: excluded
  ];
  const v = F.variableRecord(rows, 0, 2026);
  assert.strictEqual(v.n, 3);
  assert.strictEqual(v.betterWins.n, 3);
  close(v.betterWins.rate, 2 / 3, 1e-12);
  close(v.betterWins.se, Math.sqrt((2 / 3) * (1 / 3) / 3), 1e-12);
  assert.strictEqual(v.byRound.R64.n, 2);
  close(v.byRound.R64.rate, 1, 1e-12);
  assert.strictEqual(v.byRound.R32.n, 1);
  close(v.byRound.R32.rate, 0, 1e-12);
  assert.ok(!('NCG' in v.byRound), 'the displayed season must not be counted');
});

check('correlation with margin is Pearson over the same rows', () => {
  const rows = [1, 2, 3, 4].map(i => ({ y: 2000, x: [i], m: 2 * i + 1, r: 'R64' }));
  const v = F.variableRecord(rows, 0, 2026);
  close(v.corr, 1, 1e-12);
  const anti = [1, 2, 3, 4].map(i => ({ y: 2000, x: [i], m: -i, r: 'R64' }));
  close(F.variableRecord(anti, 0, 2026).corr, -1, 1e-12);
});

check('no usable rows yields null', () => {
  assert.strictEqual(F.variableRecord([{ y: 2026, x: [1], m: 1, r: 'R64' }], 0, 2026), null);
  assert.strictEqual(F.variableRecord([{ y: 2000, x: [0], m: 1, r: 'R64' }], 0, 2026), null);
});


console.log('\nmodel sensitivity (exclusion refits, preregistered)');

// Synthetic seasons: margin = 3*x0 + 2*x1 + noise, x2 = x0 + tiny noise (a
// near-duplicate), x1 independent. Enough rows for folds from 2014.
function synth(seed) {
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 - 0.5; };
  const rows = [];
  for (let y = 2010; y <= 2024; y++) {
    for (let g = 0; g < 63; g++) {
      const x0 = rnd() * 4, x1 = rnd() * 4, x2 = x0 + rnd() * 0.05;
      rows.push({ y, x: [x0, x1, x2], m: Math.round(3 * x0 + 2 * x1 + rnd() * 8) || 1, r: 'R64' });
    }
  }
  return rows;
}
const YEARS = Array.from({ length: 15 }, (_, i) => 2010 + i);

check('one exclusion model per column, each over the remaining columns', () => {
  const ex = F.exclusionModels(synth(1), [0, 1, 2], YEARS, 2025, 2014);
  assert.strictEqual(ex.length, 3);
  assert.deepStrictEqual(ex[0].cols, [1, 2]);
  assert.deepStrictEqual(ex[1].cols, [0, 2]);
  assert.deepStrictEqual(ex[2].cols, [0, 1]);
  for (const e of ex) { assert.ok(e.fit.ok); assert.strictEqual(e.fit.beta.length, 2); assert.ok(e.oos && e.oos.n > 0); }
});

check('exclusion is a REFIT: remaining coefficients move when columns are correlated', () => {
  const rows = synth(2);
  const full = F.fitLinear(rows, [0, 1, 2], 2025);
  const ex = F.exclusionModels(rows, [0, 1, 2], YEARS, 2025, 2014);
  // Without x2 (x0's near-duplicate), x0's coefficient must take up what
  // the pair used to split. Zeroing x2 would have left it at full.beta[0].
  const withoutX2 = ex[2].fit;
  assert.ok(Math.abs(withoutX2.beta[0] - full.beta[0]) > 0.5,
    `x0 coefficient did not move (${full.beta[0]} -> ${withoutX2.beta[0]}): that is zeroing, not refitting`);
});

check('absorption is real: excluding one of a duplicated pair barely changes predictions', () => {
  const rows = synth(3);
  const full = F.fitLinear(rows, [0, 1, 2], 2025);
  const ex = F.exclusionModels(rows, [0, 1, 2], YEARS, 2025, 2014);
  const withoutX2 = ex[2].fit;
  let maxDiff = 0;
  for (const r of rows.slice(0, 200)) {
    const pf = F.predictMargin(full.beta, [0, 1, 2], r.x);
    const pe = F.predictMargin(withoutX2.beta, [0, 1], r.x);
    maxDiff = Math.max(maxDiff, Math.abs(pf - pe));
  }
  assert.ok(maxDiff < 0.5, `predicted margins moved by up to ${maxDiff} after removing a near-duplicate`);
  // ...whereas removing the independent x1 changes them a lot.
  const withoutX1 = ex[1].fit;
  let big = 0;
  for (const r of rows.slice(0, 200)) {
    big = Math.max(big, Math.abs(F.predictMargin(full.beta, [0, 1, 2], r.x) - F.predictMargin(withoutX1.beta, [0, 2], r.x)));
  }
  assert.ok(big > 2, `removing an independent signal barely moved predictions (${big})`);
});

check('the training boundary is the full model\'s: no row at or after asOf is fitted', () => {
  const rows = synth(4);
  // Corrupt every row from 2020 on; a fit for asOf=2020 must be unaffected.
  const clean = F.exclusionModels(rows, [0, 1, 2], YEARS, 2020, 2014);
  const bad = rows.map(r => (r.y >= 2020 ? { ...r, m: -r.m * 50 } : r));
  const dirty = F.exclusionModels(bad, [0, 1, 2], YEARS, 2020, 2014);
  for (let j = 0; j < 3; j++) {
    assert.deepStrictEqual(dirty[j].fit.beta, clean[j].fit.beta, `column ${j}`);
    assert.strictEqual(dirty[j].oos.accuracy, clean[j].oos.accuracy);
  }
});


console.log('\nrule search (experimental strategy)');

// A 4-team "season": first_round [0,1,2,3]; two criteria. Under `a` team 0
// beats 1 and 2 beats 3, then 0 beats 2 (champion 0). Under `b` team 1 beats
// 0 and 3 beats 2, then 3 beats 1 (champion 3). Seeds break ties.
function tiny(actual) {
  return { year: 2000, first_round: [0, 1, 2, 3], seed: [1, 2, 1, 2],
           crit: { a: [4, 3, 2, 1], b: [1, 2, 3, 4] }, actual };
}
const KEYS = ['a', 'b'];

check('rulePlay follows the board tie rule: value, then better seed, then lower index', () => {
  const s = tiny([[0, 2], [0]]);
  assert.deepStrictEqual(F.rulePlay([0, 1, 2, 3], s.crit, s.seed, 'a'), [0, 2]);
  assert.deepStrictEqual(F.rulePlay([0, 1, 2, 3], s.crit, s.seed, 'b'), [1, 3]);
  const tied = { c: [5, 5, 5, 5] };
  assert.deepStrictEqual(F.rulePlay([0, 1, 2, 3], tied, [2, 1, 1, 1], 'c'), [1, 2]);   // seed, then lower index
});

check('sequences that reproduce checkpoints are exactly the ones that do', () => {
  // 4 teams = 2 rounds (r0: two games, r1: final). Checkpoint 1 = champion.
  const s = tiny([[0, 2], [0]]);
  const found = F.ruleSequencesForSeason(s, KEYS, new Set([1]), 1);
  // Brute force: 2^2 sequences; champion 0 needs a then a (a,b gives 2 vs 0 under b -> 2; b,a gives 1 vs 3 under a -> 1)
  const brute = [];
  for (const k0 of KEYS) for (const k1 of KEYS) {
    let f = F.rulePlay(s.first_round, s.crit, s.seed, k0); f = F.rulePlay(f, s.crit, s.seed, k1);
    if (f[0] === 0) brute.push(k0 + k1);
  }
  const decoded = [...found].map(c => F.decodeRule(c, KEYS, 2).join('')).sort();
  assert.deepStrictEqual(decoded, brute.sort());
  assert.deepStrictEqual(decoded, ['aa']);
});

check('ruleSearch intersects seasons and backs off to the longest surviving range', () => {
  const s1 = tiny([[0, 2], [0]]);                 // champion 0: needs a,a
  const s2 = tiny([[1, 3], [3]]);                 // champion 3: needs b,b
  const both = F.ruleSearch([s1, s2], KEYS, new Set([1]));
  // Nothing satisfies both; drops the EARLIEST (s1) and reports s2 alone.
  assert.strictEqual(both.backedOff, true);
  assert.deepStrictEqual(both.usedSeasons, [2000]);
  assert.deepStrictEqual(both.rules.map(r => r.join('')), ['bb']);
  const same = F.ruleSearch([s1, s1], KEYS, new Set([1]));
  assert.strictEqual(same.backedOff, false);
  assert.deepStrictEqual(same.rules.map(r => r.join('')), ['aa']);
});

check('rules are ranked simplest first: distinct criteria, then switches', () => {
  const seqs = [['a', 'b', 'a'], ['a', 'a', 'a'], ['a', 'a', 'b'], ['b', 'b', 'b']];
  seqs.sort((x, y) => { const cx = F.ruleComplexity(x), cy = F.ruleComplexity(y); return cx[0] - cy[0] || cx[1] - cy[1] || x.join().localeCompare(y.join()); });
  assert.deepStrictEqual(seqs.map(q => q.join('')), ['aaa', 'bbb', 'aab', 'aba']);
});

check('ruleBracket reuses the last criterion for rounds past the rule, and ruleReproduces checks checkpoints', () => {
  const s = tiny([[0, 2], [0]]);
  const rounds = F.ruleBracket(s, ['a']);           // one criterion, applied to both rounds
  assert.deepStrictEqual(rounds.map(g => g.map(x => x.win)), [[0, 2], [0]]);
  assert.strictEqual(rounds.length, 2, 'stops when one team remains');
  assert.strictEqual(F.ruleReproduces(s, ['a'], [1]), true);
  assert.strictEqual(F.ruleReproduces(s, ['b'], [1]), false);
});

console.log(`\n${passed} checks passed`);
