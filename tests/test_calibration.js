/* Checks for the probability link in docs/fit.js.
 *
 * The regression predicts a margin; the link turns it into the percentage the
 * board shows. Errors here are invisible in every margin-based metric -- RMSE,
 * MAE and accuracy are all unchanged by a monotone reparameterisation of the
 * link -- so a broken link produces a model that still looks fine on the
 * numbers the UI reports most prominently while its probabilities are wrong.
 * Hence checking it separately, against closed forms rather than against
 * itself.
 *
 * Run: node tests/test_calibration.js
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

console.log('student-t link');

check('t CDF matches published critical values', () => {
  // Each t is the 97.5th percentile for its df, so the CDF must be 0.975.
  close(F.studentTCdf(12.706, 1), 0.975, 1e-4, 'df=1');
  close(F.studentTCdf(4.303, 2), 0.975, 1e-4, 'df=2');
  close(F.studentTCdf(3.182, 3), 0.975, 1e-4, 'df=3');
  close(F.studentTCdf(2.228, 10), 0.975, 1e-4, 'df=10');
  close(F.studentTCdf(2.086, 20), 0.975, 1e-4, 'df=20');
});

check('t CDF is symmetric about zero', () => {
  for (const nu of [2, 3, 8, 30]) {
    close(F.studentTCdf(0, nu), 0.5, 1e-12, `median at df=${nu}`);
    for (const t of [0.3, 1, 2.5, 6]) {
      close(F.studentTCdf(-t, nu) + F.studentTCdf(t, nu), 1, 1e-9, `symmetry df=${nu} t=${t}`);
    }
  }
});

check('t CDF converges to the normal as df grows', () => {
  for (const t of [-2, -0.5, 0.5, 2]) {
    close(F.studentTCdf(t, 1e7), F.normalCdf(t), 1e-6, `df->inf at t=${t}`);
    close(F.studentTCdf(t, Infinity), F.normalCdf(t), 1e-12, `df=Infinity at t=${t}`);
  }
});

check('t has genuinely fatter tails than the normal', () => {
  // The whole reason for the swap: at df=3 a 4-sigma margin must stay a
  // recoverable distance from certainty, where the normal has effectively
  // pinned. Anything else and the link change bought nothing.
  const tailT = 1 - F.studentTCdf(4, 3);
  const tailN = 1 - F.normalCdf(4);
  assert.ok(tailT > 100 * tailN, `t tail ${tailT} should dwarf normal tail ${tailN}`);
  assert.ok(tailT > 1e-3, 't at df=3 must not pin at 4 sigma');
});

console.log('clipping');

check('probabilities are bounded away from 0 and 1', () => {
  for (const m of [-1e9, -50, 0, 50, 1e9]) {
    const p = F.winProbFromMargin(m, 10, { a: 1, nu: Infinity });
    assert.ok(p >= F.PROB_CLIP && p <= 1 - F.PROB_CLIP, `p=${p} out of clip bounds for m=${m}`);
    assert.ok(Number.isFinite(Math.log(p)) && Number.isFinite(Math.log(1 - p)),
      `log loss must stay finite at m=${m}`);
  }
});

check('the clip does not disturb ordinary predictions', () => {
  // A backstop that alters everyday output is a bug, not a backstop.
  for (const m of [-8, -3, 0, 3, 8]) {
    const raw = F.studentTCdf(m / 11, 3);
    close(F.winProbFromMargin(m, 11, { a: 1, nu: 3 }), raw, 1e-12, `m=${m}`);
  }
});

console.log('antisymmetry');

check('swapping the two teams flips the probability exactly', () => {
  // The model has no intercept precisely so that margin(A,B) = -margin(B,A).
  // The link must preserve that or the board contradicts itself.
  for (const cal of [{ a: 1, nu: Infinity }, { a: 1.53, nu: 3 }, { a: 2.2, nu: 2 }]) {
    for (const m of [0.5, 4, 17]) {
      const f = F.winProbFromMargin(m, 9, cal);
      const r = F.winProbFromMargin(-m, 9, cal);
      close(f + r, 1, 1e-9, `a=${cal.a} nu=${cal.nu} m=${m}`);
    }
  }
  close(F.winProbFromMargin(0, 9, { a: 1.5, nu: 3 }), 0.5, 1e-12, 'a dead heat is 50%');
});

console.log('calibration fitting');

check('calibration recovers a known scale distortion', () => {
  // Build rows whose TRUE win probability is Phi(margin / 8) but whose recorded
  // sigma is 16 -- i.e. a model understating its own confidence by exactly 2x,
  // which is the defect calibration exists to correct. It should find a ~ 2.
  const rows = [];
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 4000; i++) {
    const p = (rnd() - 0.5) * 60;                 // predicted margin
    const trueP = F.normalCdf(p / 8);
    rows.push({ m: rnd() < trueP ? 1 : -1, p, sigma: 16 });
  }
  const cal = F.calibrate(rows);
  close(cal.a, 2, 0.25, 'recovered scale');
});

check('calibration is a no-op on already-calibrated data', () => {
  const rows = [];
  let seed = 11;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 4000; i++) {
    const p = (rnd() - 0.5) * 60;
    const trueP = F.normalCdf(p / 12);
    rows.push({ m: rnd() < trueP ? 1 : -1, p, sigma: 12 });
  }
  const cal = F.calibrate(rows);
  close(cal.a, 1, 0.2, 'scale should stay near 1');
});

check('calibration never worsens log loss against the uncalibrated link', () => {
  // a=1, nu=Infinity is inside the search space, so the optimum can only be
  // at least as good. If this fails the search is broken.
  const rows = [];
  let seed = 3;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 2000; i++) {
    const p = (rnd() - 0.5) * 40;
    rows.push({ m: rnd() < F.normalCdf(p / 5) ? 1 : -1, p, sigma: 14 });
  }
  const cal = F.calibrate(rows);
  const base = F.logLossFor(rows, 1, Infinity);
  assert.ok(cal.logLoss <= base + 1e-9,
    `calibrated ${cal.logLoss} must not exceed uncalibrated ${base}`);
});

check('ties are excluded rather than scored as a loss', () => {
  // m === 0 has no winner. Scoring it either way would be inventing an outcome.
  const withTies = [
    { m: 1, p: 5, sigma: 10 }, { m: 0, p: 5, sigma: 10 }, { m: -1, p: -5, sigma: 10 },
  ];
  const noTies = [{ m: 1, p: 5, sigma: 10 }, { m: -1, p: -5, sigma: 10 }];
  // Same summed loss, differing only by the row count each divides by.
  close(F.logLossFor(withTies, 1, 3) * 3, F.logLossFor(noTies, 1, 3) * 2, 1e-12);
});


console.log('causal calibration for a displayed season (audit 2026-09, Step 2 P1-1)');

/* Synthetic matrix: 14 seasons, one variable, true margin = 8x + noise.
 * Seasons >= asOf are then CORRUPTED (margins flipped) -- a calibration that
 * looks at them will move; one that only sees strictly earlier seasons cannot.
 * This is the regression test for the page fitting (a, nu) on the season it
 * was displaying. */
function synthMatrix(corruptFrom) {
  const rows = [];
  let seed = 11;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const years = [];
  for (let y = 2010; y <= 2026; y++) {
    if (y === 2020) continue;
    years.push(y);
    for (let i = 0; i < 63; i++) {
      const x = gauss();
      let m = Math.round(8 * x + 11 * gauss());
      if (corruptFrom !== null && y >= corruptFrom) m = -m - 25;   // wildly wrong sign and scale
      rows.push({ y, x: [x], m });
    }
  }
  return { rows, years };
}

check('calibration for asOf is unchanged when seasons >= asOf are corrupted', () => {
  const clean = synthMatrix(null);
  const dirty = synthMatrix(2019);
  const a = F.causalWalkForward(clean.rows, [0], clean.years, 2019, 2014);
  const b = F.causalWalkForward(dirty.rows, [0], dirty.years, 2019, 2014);
  assert.ok(a && b, 'both should evaluate');
  close(a.calibration.a, b.calibration.a, 1e-12, 'scale a');
  assert.strictEqual(a.calibration.nu, b.calibration.nu, 'tail nu');
  assert.strictEqual(a.n, b.n, 'held-out row count');
  assert.ok(a.n === 63 * 5, `only 2014-2018 are held out for 2019, got ${a.n} rows`);
});

check('the old global crossValidate DOES move under the same corruption (the defect)', () => {
  // Documents why the helper exists: this is the behaviour app.js had.
  const clean = synthMatrix(null);
  const dirty = synthMatrix(2019);
  const a = F.crossValidate(clean.rows, [0], clean.years, 2014);
  const b = F.crossValidate(dirty.rows, [0], dirty.years, 2014);
  assert.ok(Math.abs(a.calibration.a - b.calibration.a) > 0.05 || a.calibration.nu !== b.calibration.nu,
    'global calibration should be contaminated by later seasons; if not, the fixture is too weak');
});

check('scale is shrunk toward 1 with weight n/(n+63)', () => {
  const { rows, years } = synthMatrix(null);
  const o = F.causalWalkForward(rows, [0], years, 2026, 2014);
  const w = o.n / (o.n + F.CAL_PRIOR_STRENGTH);
  close(o.calibration.a, w * o.calibrationRaw.a + (1 - w), 1e-12, 'shrunk a');
  assert.strictEqual(o.calibration.nu, o.calibrationRaw.nu);
});

check('a season with no earlier held-out folds yields null rather than a leaked calibration', () => {
  const { rows, years } = synthMatrix(null);
  assert.strictEqual(F.causalWalkForward(rows, [0], years, 2014, 2014), null);
});

check('app.js refit() uses the causal helper (guards against reverting to crossValidate)', () => {
  const fs = require('fs');
  const src = fs.readFileSync(path.join(__dirname, '..', 'docs', 'app.js'), 'utf8');
  const refit = src.slice(src.indexOf('function refit()'), src.indexOf('function diffVector'));
  assert.ok(/causalWalkForward\(state\.training\.games,\s*cols,\s*state\.training\.years,\s*state\.year/.test(refit),
    'refit() must calibrate via causalWalkForward(..., state.year, ...)');
  const code = refit.split('\n').filter(l => !/^\s*(\/\/|\*|\/\*)/.test(l)).join('\n');
  assert.ok(!/crossValidate\(/.test(code), 'refit() must not call crossValidate directly');
});

console.log(`\n${passed} checks passed`);
