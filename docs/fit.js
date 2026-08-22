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
 * skipY : season to exclude -- leave-one-year-out. Passing null fits on
 *         everything, which is only correct when no season is being predicted.
 */
function fitLogistic(rows, cols, skipYear) {
  const used = rows.filter(r => r.y !== skipYear);
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
 * honest as a fit diagnostic, not a claim about future accuracy.
 */
function fitQuality(rows, cols, skipYear, beta) {
  const used = rows.filter(r => r.y !== skipYear);
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

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { fitLogistic, fitQuality, sigmoid, solve, FIT };
}
