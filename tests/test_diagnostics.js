/* Checks for docs/diag.js.
 *
 * The diagnostics tab makes statistical claims -- p-values, VIFs, an F test --
 * and a wrong one is worse than none, because it looks authoritative. Every
 * quantity below is checked against a value computed independently: published
 * critical values for the distributions, and closed-form results for the
 * regression pieces.
 *
 * Run: node tests/test_diagnostics.js
 */

const assert = require('assert');
const path = require('path');
const D = require(path.join(__dirname, '..', 'docs', 'diag.js'));

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

console.log('distributions');

check('t two-sided p matches published critical values', () => {
  // Each t is the 97.5th percentile for its df, so two-sided p must be 0.05.
  close(D.tPValue(2.228, 10), 0.05, 1e-4);
  close(D.tPValue(2.086, 20), 0.05, 1e-4);
  close(D.tPValue(1.984, 100), 0.05, 1e-4);
  close(D.tPValue(1.9600, 1e6), 0.05, 1e-4);
});

check('t p-value is symmetric and monotone', () => {
  close(D.tPValue(2.5, 50), D.tPValue(-2.5, 50), 1e-12, 'symmetry');
  assert.ok(D.tPValue(3, 50) < D.tPValue(2, 50), 'larger |t| must give smaller p');
  close(D.tPValue(0, 50), 1, 1e-9, 'p at t=0');
});

check('chi-square(1) upper tail matches published values', () => {
  close(D.chi2P1(3.841), 0.05, 1e-3);
  close(D.chi2P1(6.635), 0.01, 1e-3);
  close(D.chi2P1(2.706), 0.10, 1e-3);
});

check('incomplete beta hits known closed forms', () => {
  close(D.betai(1, 1, 0.37), 0.37, 1e-9, 'I_x(1,1) = x');
  close(D.betai(2, 1, 0.5), 0.25, 1e-9, 'I_x(2,1) = x^2');
  close(D.betai(0.5, 0.5, 0.5), 0.5, 1e-9, 'symmetric at midpoint');
});

console.log('linear algebra');

check('matrix inverse is correct and detects singularity', () => {
  const inv = D.matInverse([[4, 7], [2, 6]]);
  close(inv[0][0], 0.6, 1e-12);
  close(inv[0][1], -0.7, 1e-12);
  close(inv[1][0], -0.2, 1e-12);
  close(inv[1][1], 0.4, 1e-12);
  assert.strictEqual(D.matInverse([[1, 2], [2, 4]]), null, 'singular must return null, never a guess');
});

console.log('regression');

/* A dataset with an exactly known answer: y = 3*x1 - 2*x2 with no noise.
 * Coefficients must be recovered exactly and residuals must vanish. */
function exactRows() {
  const rows = [];
  for (let i = 0; i < 60; i++) {
    const x1 = Math.sin(i) * 2, x2 = Math.cos(i * 1.7);
    rows.push({ y: 2000, x: [x1, x2], m: 3 * x1 - 2 * x2 });
  }
  return rows;
}

check('OLS recovers exact coefficients on noiseless data', () => {
  const f = D.ols(exactRows(), [0, 1], null, false);
  close(f.beta[0], 3, 1e-9, 'beta1');
  close(f.beta[1], -2, 1e-9, 'beta2');
  close(f.rss, 0, 1e-16, 'residual sum of squares');
  close(f.r2_about_zero, 1, 1e-12, 'R2');
});

check('an orthogonal design has VIF exactly 1', () => {
  // Two columns constructed orthogonal: VIF must be 1 for both.
  const rows = [];
  for (let i = 0; i < 40; i++) {
    const a = i % 2 === 0 ? 1 : -1;
    const b = i % 4 < 2 ? 1 : -1;      // orthogonal to a over this range
    rows.push({ y: 2000, x: [a, b], m: a + b });
  }
  const f = D.ols(rows, [0, 1], null, false);
  const v = D.vifTable(f, ['a', 'b']);
  close(v[0].vif, 1, 1e-9, 'VIF a');
  close(v[1].vif, 1, 1e-9, 'VIF b');
});

check('VIF rises with collinearity and flags severity', () => {
  const rows = [];
  for (let i = 0; i < 200; i++) {
    const a = Math.sin(i);
    const b = a + 0.02 * Math.cos(i * 3.1);   // nearly a duplicate of a
    rows.push({ y: 2000, x: [a, b], m: a + Math.sin(i * 0.3) });
  }
  const f = D.ols(rows, [0, 1], null, false);
  const v = D.vifTable(f, ['a', 'b']);
  assert.ok(v[0].vif > 10, `near-duplicate column should have high VIF, got ${v[0].vif}`);
  assert.strictEqual(v[0].severity, 'severe');
});

check('perfect collinearity is reported, not silently fitted', () => {
  const rows = [];
  for (let i = 0; i < 50; i++) {
    const a = Math.sin(i);
    rows.push({ y: 2000, x: [a, 2 * a], m: a });   // exact duplicate, scaled
  }
  const r = D.diagnose(rows, [0, 1], ['a', 'double_a'], null);
  assert.ok(r.error && /collinear/i.test(r.error), 'must return an error, got: ' + JSON.stringify(r.error));
});

check('standard errors shrink as the sample grows', () => {
  const make = n => {
    const rows = [];
    for (let i = 0; i < n; i++) {
      const x = Math.sin(i * 0.7);
      rows.push({ y: 2000, x: [x], m: 2 * x + Math.sin(i * 13.1) });   // deterministic pseudo-noise
    }
    return rows;
  };
  const small = D.coefficientTable(D.ols(make(50), [0], null, false), ['x']);
  const big = D.coefficientTable(D.ols(make(800), [0], null, false), ['x']);
  assert.ok(big[0].se < small[0].se, 'more data must give a tighter standard error');
});

check('a pure-noise predictor is not called significant', () => {
  // x alternates deterministically and is uncorrelated with the target.
  const rows = [];
  for (let i = 0; i < 400; i++) {
    rows.push({ y: 2000, x: [i % 2 ? 1 : -1], m: Math.sin(i * 2.399963) * 10 });
  }
  const t = D.coefficientTable(D.ols(rows, [0], null, false), ['noise']);
  assert.ok(t[0].p > 0.05, `unrelated predictor should not reach p<0.05, got p=${t[0].p}`);
});

check('a strong predictor is called significant', () => {
  const rows = [];
  for (let i = 0; i < 400; i++) {
    const x = Math.sin(i * 0.7);
    rows.push({ y: 2000, x: [x], m: 5 * x + 0.3 * Math.sin(i * 13.1) });
  }
  const t = D.coefficientTable(D.ols(rows, [0], null, false), ['strong']);
  assert.ok(t[0].significant && t[0].p < 1e-6, `strong signal should be significant, got p=${t[0].p}`);
});

console.log('intercept and residuals');

check('a real intercept is detected', () => {
  const rows = [];
  for (let i = 0; i < 300; i++) {
    const x = Math.sin(i * 0.7);
    rows.push({ y: 2000, x: [x], m: 4 + 2 * x + 0.2 * Math.sin(i * 13.1) });   // constant of 4
  }
  const it = D.interceptTest(rows, [0], null, ['x']);
  close(it.intercept, 4, 0.2, 'recovered intercept');
  assert.ok(it.significant, 'a constant of 4 must test significant');
});

check('an absent intercept is not invented', () => {
  const rows = [];
  for (let i = 0; i < 300; i++) {
    const x = Math.sin(i * 0.7);
    rows.push({ y: 2000, x: [x], m: 2 * x + 0.2 * Math.sin(i * 13.1) });
  }
  const it = D.interceptTest(rows, [0], null, ['x']);
  assert.ok(!it.significant, `no constant present, but test said significant (p=${it.p})`);
});

/* REGRESSION GUARD for a bug that shipped in the first version of this file.
 *
 * The test used to regress squared residuals on the SIGNED fitted value. This
 * model is antisymmetric -- swapping the teams negates the prediction and
 * leaves the error spread alone -- so variance is an even function of the
 * fitted value and the signed slope is zero by construction. The test could not
 * detect the only kind of heteroscedasticity the design admits: it returned
 * p = 0.92 on data whose error was explicitly scaled by |x|.
 *
 * The data below is built symmetric on purpose. Any implementation that keys
 * off the signed fitted value fails it; only one keyed off the magnitude
 * passes. */
check('heteroscedasticity is detected when present and not when absent', () => {
  const homo = [], hetero = [];
  for (let i = 0; i < 500; i++) {
    const x = Math.sin(i * 0.7) * 3;
    const noise = Math.sin(i * 13.1);
    homo.push({ y: 2000, x: [x], m: 2 * x + noise });
    // Error scaled by |x|: spread grows with the SIZE of the prediction, and is
    // symmetric about zero, exactly as an antisymmetric model requires.
    hetero.push({ y: 2000, x: [x], m: 2 * x + noise * Math.abs(x) * 3 });
  }
  const a = D.residualDiagnostics(D.ols(homo, [0], null, false));
  const b = D.residualDiagnostics(D.ols(hetero, [0], null, false));
  assert.ok(!a.heteroscedasticity.significant, `constant variance flagged (p=${a.heteroscedasticity.p})`);
  assert.ok(b.heteroscedasticity.significant, `scaled variance missed (p=${b.heteroscedasticity.p})`);
  assert.ok(b.heteroscedasticity.widensWithMargin, 'spread grows with |prediction|, must be reported as such');
});

check('the variance test is blind to sign, as antisymmetry requires', () => {
  // Same data, every row mirrored: (x, m) -> (-x, -m). A correct test must
  // return an identical verdict, because the two orderings describe the same
  // games. The old signed implementation was sensitive to this.
  const rows = [], mirrored = [];
  for (let i = 0; i < 500; i++) {
    const x = Math.sin(i * 0.7) * 3;
    const m = 2 * x + Math.sin(i * 13.1) * Math.abs(x) * 3;
    rows.push({ y: 2000, x: [x], m });
    mirrored.push({ y: 2000, x: [-x], m: -m });
  }
  const a = D.residualDiagnostics(D.ols(rows, [0], null, false));
  const b = D.residualDiagnostics(D.ols(mirrored, [0], null, false));
  close(a.heteroscedasticity.statistic, b.heteroscedasticity.statistic, 1e-9,
        'statistic must be invariant to row orientation');
});

check('residuals are mean-zero and outliers are counted', () => {
  const rows = exactRows();
  rows.push({ y: 2000, x: [0, 0], m: 400 });    // one absurd game
  const res = D.residualDiagnostics(D.ols(rows, [0, 1], null, false));
  assert.ok(res.outlierCount >= 1, 'the planted outlier must be caught');
  assert.ok(res.points.length === rows.length, 'every row must appear in the plot');
});

console.log('baselines');

check('baselines are ordered sensibly and the fit beats predicting zero', () => {
  const rows = [];
  for (let i = 0; i < 300; i++) {
    const x = Math.sin(i * 0.7);
    rows.push({ y: 2000, x: [x], m: 6 * x + 0.5 * Math.sin(i * 13.1) });
  }
  const f = D.ols(rows, [0], null, false);
  const b = D.baselines(rows, [0], null, f);
  const zero = b[0], fitted = b[b.length - 1];
  assert.ok(fitted.rmse < zero.rmse, 'a real model must beat predicting a dead heat');
  close(zero.r2, 0, 1e-12, 'R2 of the zero baseline is 0 by definition');
});

console.log(`\n${passed} checks passed`);
