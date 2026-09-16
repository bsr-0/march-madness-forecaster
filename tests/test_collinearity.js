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

console.log(`\n${passed} checks passed`);
