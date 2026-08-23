/* Live spread regression over historical tournament games.
 *
 * The user enables variables; this fits the coefficients. That is the whole
 * point of the design: nobody has to assert whether more freshman minutes helps,
 * because the data answers it and the answer is shown.
 *
 * MODEL
 *   predicted margin = B . (z1 - z2)          [points]
 *   P(team1 wins)    = Phi(predicted margin / sigma)
 *
 * z is standardised within its own season, so a coefficient reads directly as
 * "points of margin per standard deviation of edge in this variable" -- a unit
 * anyone can check against intuition. sigma is the residual spread of the fit,
 * which is what turns a predicted margin into a probability.
 *
 * WHY MARGIN AND NOT WIN/LOSS. A 1-point escape and a 30-point demolition are
 * the same event to a classifier and very different evidence about the teams.
 * Fitting margin uses that, and RMSE/MAE/R2 become available as evaluation
 * metrics. The bracket still needs a winner, and gets one for free: predicted
 * margin > 0 is the same statement as predicted win, so accuracy is still
 * reported and is still what the board is graded on.
 *
 * NO INTERCEPT, deliberately. The rows are differentials, so the model must be
 * antisymmetric: swapping the two teams has to flip the predicted margin
 * exactly. A free constant would let it learn "the team written first tends to
 * win", which is an artefact of row order rather than basketball. Fixing the
 * intercept at 0 is what makes margin(A,B) = -margin(B,A) hold by construction.
 *
 * Mirroring every row to (-x, -m) would enforce the same thing, and it is what
 * the payload's fitting contract describes. It is skipped here because for a
 * zero-intercept fit the mirrored rows contribute an identical normal equation
 * -- they exactly double both X'X and X'y -- so they change nothing but cost.
 *
 * RIDGE, AND WHAT IT DOES NOT FIX. Several variables are near-collinear:
 * overall rating and national rank are near-substitutes, and both are largely
 * functions of offense and defense. The L2 penalty here is deliberately light,
 * because tightening it costs real accuracy -- measured on the full 26-variable
 * set, walk-forward:
 *
 *     ridge per 1k rows     accuracy   max |coefficient|   sign-flipping vars
 *     1  (shipped)            78.2%          40.7               14 of 26
 *     20                      73.7%          15.3                9
 *     400                     73.9%           3.0                4
 *
 * So the shipped model predicts best and reads worst. With all 26 enabled it
 * will print something like "-39 x rating + 36 x national rank": two
 * near-identical variables handed enormous offsetting coefficients. The SUM is
 * stable and predicts well; the individual numbers are not, and more than half
 * of them change sign between folds.
 *
 * This is not hidden. `stability()` measures it per coefficient and the
 * equation marks the unstable ones, because a displayed weight that cannot be
 * interpreted should say so rather than look authoritative. Enabling fewer,
 * less redundant variables gives coefficients that mean what they appear to
 * mean, at a small cost in accuracy.
 */

const FIT = {
  LAMBDA: 1.0,        // ridge strength per 1,000 rows, in standardised units
  MIN_ROWS_PER_COL: 5,
  MIN_TEST_YEAR: 2014,
};

/* Standard normal CDF. Zelen & Severo 26.2.17 -- error below 7.5e-8, which is
 * four orders of magnitude finer than anything displayed. */
function normalCdf(t) {
  const s = t < 0 ? -1 : 1;
  const x = Math.abs(t) / Math.SQRT2;
  const k = 1 / (1 + 0.3275911 * x);
  const poly = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741 + k * (-1.453152027 + k * 1.061405429))));
  return 0.5 * (1 + s * (1 - poly * Math.exp(-x * x)));
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

function trainingRows(rows, asOf) {
  return asOf === null || asOf === undefined ? rows : rows.filter(r => r.y < asOf);
}

/* Fit predicted margin by ridge least squares, solved in one step.
 *
 * rows  : [{x: number[], m: number}]  full-width differentials and margins
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
 *
 * `sigma` comes back with the coefficients because a margin alone cannot fill a
 * bracket -- the board needs P(win), and sigma is what converts one to the
 * other. It is the RMS training residual, i.e. how wrong this model typically
 * is in points.
 */
function fitLinear(rows, cols, asOf) {
  const used = trainingRows(rows, asOf);
  const k = cols.length;
  if (!k || used.length < k * FIT.MIN_ROWS_PER_COL) {
    return { beta: cols.map(() => 0), sigma: 1, n: used.length, ok: false, reason: 'not enough data', cols };
  }

  const A = Array.from({ length: k }, () => new Array(k).fill(0));
  const b = new Array(k).fill(0);

  for (const r of used) {
    for (let a = 0; a < k; a++) {
      const xa = r.x[cols[a]];
      b[a] += xa * r.m;
      for (let c = a; c < k; c++) A[a][c] += xa * r.x[cols[c]];
    }
  }
  // A is symmetric; only the upper triangle was accumulated.
  const ridge = FIT.LAMBDA * (used.length / 1000);
  for (let a = 0; a < k; a++) {
    for (let c = a; c < k; c++) A[c][a] = A[a][c];
    A[a][a] += ridge;
  }

  const beta = solve(A, b);
  if (!beta) {
    return { beta: cols.map(() => 0), sigma: 1, n: used.length, ok: false, reason: 'singular', cols };
  }

  let sse = 0;
  for (const r of used) {
    let p = 0;
    for (let a = 0; a < k; a++) p += beta[a] * r.x[cols[a]];
    sse += (r.m - p) ** 2;
  }
  // Residual spread, floored so a degenerate fit cannot produce infinite
  // confidence from a zero denominator.
  const sigma = Math.max(Math.sqrt(sse / used.length), 1e-6);

  return { beta, sigma, n: used.length, ok: true, cols };
}

function predictMargin(beta, cols, x) {
  let p = 0;
  for (let a = 0; a < cols.length; a++) p += beta[a] * x[cols[a]];
  return p;
}

/* Error metrics for a set of rows against a fitted model.
 *
 * R2 IS MEASURED ABOUT ZERO, NOT ABOUT THE MEAN, and that is not a detail.
 * A model with no intercept is claiming "these two teams differ by this many
 * points"; its null is "they are even", i.e. predict 0. Scoring against the
 * sample mean margin instead would define the baseline as "the first-listed
 * team wins by the average amount" -- a baseline that requires knowing which
 * team to list first, which is exactly the thing being predicted. Rows are
 * oriented by seed for this reason, but even so, mean-centred R2 would be
 * answering a question nobody asked.
 */
function scoreSpread(rows, beta, cols) {
  if (!rows.length || !cols.length) return null;
  let sse = 0, sae = 0, sst = 0, correct = 0, decided = 0;
  for (const r of rows) {
    const p = predictMargin(beta, cols, r.x);
    const e = r.m - p;
    sse += e * e;
    sae += Math.abs(e);
    sst += r.m * r.m;          // about zero -- see above
    if (r.m !== 0) {
      decided++;
      if ((p > 0) === (r.m > 0)) correct++;
    }
  }
  const n = rows.length;
  return {
    n,
    rmse: Math.sqrt(sse / n),
    mae: sae / n,
    r2: sst > 0 ? 1 - sse / sst : null,
    accuracy: decided ? correct / decided : null,
  };
}

/* In-sample fit quality on the training seasons.
 *
 * Reported so the user can see that enabling more variables does not
 * automatically mean a better model. It is IN-SAMPLE -- a fit diagnostic, not a
 * claim about future accuracy. The out-of-sample number is computed separately
 * and shipped alongside.
 */
function fitQuality(rows, cols, asOf, beta) {
  const used = trainingRows(rows, asOf);
  if (!used.length || !cols.length) return null;
  return scoreSpread(used, beta, cols);
}

/* Walk-forward out-of-sample evaluation.
 *
 * For each test season: fit on strictly earlier seasons, then score that
 * season's games, which the fit has never seen. This is the only number here
 * that says anything about how the chosen variables would do on a tournament
 * that has not happened.
 *
 * It is recomputed live because it depends on which variables are enabled, and
 * there are 2^26 possible selections. Cost is one solve per test season.
 *
 * Seasons before MIN_TEST_YEAR are not tested: with only a season or two of
 * history the fit is too thin to be a fair test of anything.
 *
 * PER-FOLD COEFFICIENTS ARE RETAINED. Each fold already fits a model; keeping
 * its coefficients costs nothing and answers a question a single full-history
 * regression cannot: is this variable's effect stable, or does it swing sign
 * between folds? A coefficient that reads +0.8, -0.2, +1.4, -0.6 is not
 * something to interpret, however good its full-sample p-value looks.
 */
function crossValidate(rows, cols, years, minYear) {
  if (!cols.length) return null;
  const testYears = years.filter(y => y >= (minYear || FIT.MIN_TEST_YEAR));
  const perYear = {};
  const trajectory = cols.map(() => []);
  const pooled = [];
  let sigmaSum = 0, folds = 0;

  for (const y of testYears) {
    const test = rows.filter(r => r.y === y);
    if (!test.length) continue;
    const f = fitLinear(rows, cols, y);
    if (!f.ok) continue;   // too little history to judge

    perYear[y] = scoreSpread(test, f.beta, cols);
    perYear[y].beta = f.beta.slice();
    f.beta.forEach((b, i) => trajectory[i].push(b));
    for (const r of test) pooled.push({ x: r.x, m: r.m, p: predictMargin(f.beta, cols, r.x) });
    sigmaSum += f.sigma;
    folds++;
  }
  if (!pooled.length) return null;

  // Aggregate over every held-out game at once rather than averaging per-season
  // figures, so a season is not weighted the same as a play-in-shortened one.
  let sse = 0, sae = 0, sst = 0, correct = 0, decided = 0;
  for (const r of pooled) {
    const e = r.m - r.p;
    sse += e * e; sae += Math.abs(e); sst += r.m * r.m;
    if (r.m !== 0) { decided++; if ((r.p > 0) === (r.m > 0)) correct++; }
  }
  const n = pooled.length;

  return {
    n,
    seasons: folds,
    rmse: Math.sqrt(sse / n),
    mae: sae / n,
    r2: sst > 0 ? 1 - sse / sst : null,
    accuracy: decided ? correct / decided : null,
    sigma: sigmaSum / folds,
    perYear,
    stability: stability(trajectory),
  };
}

/* Per-coefficient summary across the walk-forward folds.
 *
 * `signFlips` is the one to read first: a variable whose coefficient changes
 * sign between folds has no stable relationship with margin, whatever its
 * average says.
 */
function stability(trajectory) {
  return trajectory.map(series => {
    if (!series.length) return null;
    const mean = series.reduce((a, b) => a + b, 0) / series.length;
    const varr = series.reduce((a, b) => a + (b - mean) ** 2, 0) / series.length;
    const pos = series.filter(b => b > 0).length;
    return {
      mean,
      sd: Math.sqrt(varr),
      min: Math.min(...series),
      max: Math.max(...series),
      signFlips: pos !== 0 && pos !== series.length,
      series,
    };
  });
}

/* P(team1 wins), from a predicted margin and the fit's residual spread. */
function winProbFromMargin(margin, sigma) {
  return normalCdf(margin / Math.max(sigma, 1e-6));
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    fitLinear, fitQuality, crossValidate, scoreSpread, predictMargin,
    winProbFromMargin, normalCdf, solve, stability, FIT,
  };
}
