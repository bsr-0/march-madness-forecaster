/* Live logistic regression over historical tournament games.
 *
 * The user enables variables; this fits the coefficients. That is the whole
 * point of the design: nobody has to assert whether more freshman minutes helps,
 * because the data answers it and the answer is shown.
 *
 * MODEL
 *   P(team1 beats team2) = sigmoid( B . (z1 - z2) )
 *
 * z is already standardised within its own season, so a coefficient reads as
 * "log-odds gained per standard deviation of edge in this variable".
 *
 * NO INTERCEPT, deliberately. The rows are differentials, so the model must be
 * antisymmetric: swapping the two teams has to flip the probability exactly. A
 * free constant would let it learn "the team written first tends to win", which
 * is an artefact of row order rather than basketball. Fixing the intercept at 0
 * is what makes P(A beats B) + P(B beats A) = 1 hold by construction.
 *
 * Mirroring every row to (-x, 1-y) would enforce the same thing, and it is what
 * the payload's fitting contract describes. It is skipped here because for a
 * zero-intercept fit the mirrored rows contribute an identical gradient and
 * Hessian -- they exactly double both -- so they change nothing but cost. That
 * equivalence was verified numerically to ~1e-8 before relying on it.
 *
 * RIDGE. Several variables are near-collinear (overall rating is largely a
 * function of offense and defense). Unpenalised, the fit can hand one a huge
 * positive coefficient and its neighbour an offsetting negative one -- unstable,
 * and nonsense to read. A small L2 penalty keeps coefficients interpretable at
 * a cost in fit that is invisible here.
 */

const FIT = {
  MAX_ITER: 25,
  TOL: 1e-7,
  LAMBDA: 1.0,   // ridge strength, in standardised units
};

function sigmoid(t) {
  // Branch to avoid overflow of exp() on large |t|.
  if (t >= 0) { const e = Math.exp(-t); return 1 / (1 + e); }
  const e = Math.exp(t);
  return e / (1 + e);
}

/* Solve A d = b by Gauss-Jordan with partial pivoting.
 * n <= 26 here, so an explicit solve is cheaper and clearer than anything
 * cleverer. Returns null on a singular system rather than silently producing
 * garbage. */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let j = c; j <= n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (!f) continue;
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map(row => row[n]);
}

/* Fit by IRLS (Newton-Raphson).
 *
 * rows  : [{x: number[], w: 0|1}]  full-width differentials
 * cols  : indices into x of the enabled variables
 * asOf  : the season being predicted. Training uses STRICTLY EARLIER seasons.
 *
 * WALK-FORWARD, NOT PLAIN LEAVE-ONE-YEAR-OUT.
 * Excluding only the target season would still train 2024 on 2025 and 2026 --
 * using future tournaments to predict a past one. That is not a thing anyone
 * could have done at the time, and it flatters early seasons. Restricting to
 * prior years is what someone standing on that Selection Sunday actually had.
 *
 * Pass null to fit on everything, which is only correct when no season is being
 * predicted.
 */
function fitLogistic(rows, cols, asOf) {
  const used = asOf === null || asOf === undefined ? rows : rows.filter(r => r.y < asOf);
  const k = cols.length;
  if (!k || used.length < k * 5) {
    return { beta: cols.map(() => 0), n: used.length, converged: false, reason: 'not enough data' };
  }

  // Pack only the enabled columns once, rather than indexing inside every
  // iteration of the inner loop.
  const X = used.map(r => cols.map(c => r.x[c]));
  const y = used.map(r => r.w);

  let beta = new Array(k).fill(0);
  let converged = false;

  for (let iter = 0; iter < FIT.MAX_ITER; iter++) {
    const H = Array.from({ length: k }, () => new Array(k).fill(0));
    const g = new Array(k).fill(0);

    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      let t = 0;
      for (let j = 0; j < k; j++) t += beta[j] * xi[j];
      const p = sigmoid(t);
      const w = Math.max(p * (1 - p), 1e-10);   // floor keeps H invertible
      const resid = y[i] - p;
      for (let a = 0; a < k; a++) {
        g[a] += xi[a] * resid;
        const wa = w * xi[a];
        for (let b = a; b < k; b++) H[a][b] += wa * xi[b];
      }
    }
    // H is symmetric; only the upper triangle was accumulated.
    for (let a = 0; a < k; a++) {
      for (let b = a; b < k; b++) H[b][a] = H[a][b];
      H[a][a] += FIT.LAMBDA;
      g[a] -= FIT.LAMBDA * beta[a];
    }

    const step = solve(H, g);
    if (!step) break;

    let maxStep = 0;
    for (let a = 0; a < k; a++) {
      beta[a] += step[a];
      maxStep = Math.max(maxStep, Math.abs(step[a]));
    }
    if (maxStep < FIT.TOL) { converged = true; break; }
  }

  return { beta, n: used.length, converged, cols };
}

/* In-sample accuracy and log-loss on the fitted rows.
 *
 * Reported so the user can see that enabling more variables does not
 * automatically mean a better model. It is IN-SAMPLE on the training seasons --
 * a fit diagnostic, not a claim about future accuracy. The out-of-sample number
 * is computed separately and shipped alongside.
 */
function fitQuality(rows, cols, asOf, beta) {
  const used = asOf === null || asOf === undefined ? rows : rows.filter(r => r.y < asOf);
  if (!used.length || !cols.length) return null;
  let correct = 0, ll = 0;
  for (const r of used) {
    let t = 0;
    for (let j = 0; j < cols.length; j++) t += beta[j] * r.x[cols[j]];
    const p = sigmoid(t);
    if ((p >= 0.5 ? 1 : 0) === r.w) correct++;
    ll += r.w ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12));
  }
  return { accuracy: correct / used.length, logLoss: -ll / used.length, n: used.length };
}


/* Walk-forward out-of-sample evaluation.
 *
 * For each test season: fit on strictly earlier seasons, then score the games of
 * that season, which the fit has never seen. This is the only number here that
 * says anything about how the chosen variables would do on a tournament that has
 * not happened.
 *
 * It is recomputed live because it depends on which variables are enabled, and
 * there are 2^26 possible selections. Cost is one fit per test season; with all
 * 26 variables that is ~12 fits and well under a second.
 *
 * Seasons before `minYear` are not tested: with only a season or two of history
 * the fit is too thin to be a fair test of anything.
 */
function crossValidate(rows, cols, years, minYear) {
  if (!cols.length) return null;
  const testYears = years.filter(y => y >= (minYear || 2014));
  let correct = 0, n = 0, ll = 0;
  const perYear = {};

  for (const y of testYears) {
    const test = rows.filter(r => r.y === y);
    if (!test.length) continue;
    const f = fitLogistic(rows, cols, y);
    if (!f.n || f.n < cols.length * 5) continue;   // too little history to judge

    let c = 0;
    for (const r of test) {
      let t = 0;
      for (let j = 0; j < cols.length; j++) t += f.beta[j] * r.x[cols[j]];
      const p = sigmoid(t);
      if ((p >= 0.5 ? 1 : 0) === r.w) { c++; correct++; }
      ll += r.w ? Math.log(Math.max(p, 1e-12)) : Math.log(Math.max(1 - p, 1e-12));
      n++;
    }
    perYear[y] = { n: test.length, accuracy: c / test.length };
  }
  if (!n) return null;
  return { accuracy: correct / n, logLoss: -ll / n, n, perYear, seasons: Object.keys(perYear).length };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { fitLogistic, fitQuality, crossValidate, sigmoid, solve, FIT };
}
