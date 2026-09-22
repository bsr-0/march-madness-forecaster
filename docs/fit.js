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

/* ---------------------------------------------------------------- calibration
 *
 * THE LINK IS PART OF THE MODEL, NOT A FORMALITY. The regression predicts a
 * margin; something has to carry that margin to P(win). Two separate defects
 * lived in that step, and they need two separate fixes -- rescaling cannot fix
 * saturation and a fatter tail cannot fix scale.
 *
 * 1. SCALE. sigma was the IN-SAMPLE RMS residual, which understates
 *    out-of-sample error, and ridge shrinks predicted margins toward zero on
 *    top of that. Measured walk-forward over 756 held-out games, the model was
 *    systematically UNDER-confident through the 0.6-0.9 band:
 *
 *        bin        n   predicted   actual    gap    gap/SE
 *        0.6-0.7  117       64.9%    73.5%   +8.6      1.94
 *        0.7-0.8  124       75.0%    81.5%   +6.5      1.66
 *        0.8-0.9  122       84.8%    91.8%   +7.0      2.15
 *
 *    READ THAT TABLE CORRECTLY. No single bin clears two sigma by much; the
 *    evidence is that three ADJACENT bins all miss in the same direction, not
 *    any one of them. And the 0.5-0.6 bin's -6.3 point gap is NOT a finding --
 *    at n=84 and p~0.5 its standard error is 5.4 points, so it sits 1.16 SE
 *    from zero and points the opposite way from its neighbours, which is what
 *    noise looks like. It is recorded here so nobody later cites it as an
 *    S-curve.
 *
 *    That warning earned itself immediately. After calibration the same bin
 *    reads -15.1 points, -2.40 SE, which looks alarming until the bin edge is
 *    moved: [0.50,0.60) gives -2.40 SE, [0.52,0.62) gives -1.56, [0.54,0.64)
 *    gives -1.06. A real miscalibration does not care where the boundary falls;
 *    this one does, because a handful of coin-flip games crossing an arbitrary
 *    line is the whole effect. Judged on windows that do not depend on that
 *    choice: the wide [0.45,0.65) block is -0.72 SE over 117 games, and across
 *    all 756 held-out games the model expects 536.7 wins and observes 541,
 *    +0.40 SE. DO NOT tune against this bin.
 *
 *    Fix: one free parameter `a` in link(a * margin / sigma), fitted by
 *    minimising LOG LOSS. Log loss, not margin MSE, because probability is what
 *    the bracket is scored on -- the regression already optimised MSE and that
 *    is a different objective.
 *
 * 2. SATURATION, which is a LINK problem and survives any rescaling. The normal
 *    CDF has very thin tails: Phi(4) ~ 0.99997, Phi(6) ~ 1 - 1e-9. The shipped
 *    model already put 10 of 756 held-out predictions past 1e-4 of 0 or 1, the
 *    most extreme at p = 0.9999999977. None of them happened to lose, which is
 *    luck rather than safety: a 1-seed over a 16-seed is roughly a 1.3% upset
 *    historically (2 in ~156), so certainty at that level is wrong on the
 *    merits, and under log loss one miss at a pinned probability is unbounded.
 *
 *    Fix: Student-t link with the degrees of freedom fitted alongside `a`. It
 *    keeps every property that made margin regression the right choice -- still
 *    a margin, still antisymmetric, still monotone -- and fattens the tail by
 *    exactly as much as the held-out games support, rather than by assertion.
 *    nu = Infinity recovers the normal exactly, so the old behaviour remains
 *    reachable and is chosen only if the data prefers it.
 *
 * PROB_CLIP is a backstop under both, not a substitute for either. It exists so
 * that a pathological fit cannot produce a literal 0 or 1 and an infinite
 * score.
 *
 * IT DOES BIND, contrary to what this comment claimed until 2026-08-26. Whether
 * it binds depends entirely on nu. Measured per year on the walk-forward
 * calibration: with nu = 2-3 the closest any prediction comes to the bound is
 * 4.9e-3 to 2.9e-2, comfortably clear. With nu = Infinity (2014, the cold-start
 * fallback) it is 1.3e-4, past the clip, and with nu = 12 (2015) it is 9.5e-4,
 * also past. Thin tails saturate; fat tails do not. That is the Student-t doing
 * the job it was added for, and it is visible in the saturation behaviour even
 * though the mean-log-loss difference against the normal is not statistically
 * distinguishable (paired bootstrap 95% CI [-0.0032, +0.0159] on 630 games).
 *
 * HONEST ACCOUNTING, AND THE FIX. `calibrate()` here fits `a` and `nu` on
 * whatever rows it is handed, so calling it on the pooled walk-forward
 * residuals makes the resulting log loss mildly optimistic -- two parameters
 * informed by the same held-out games they are then scored on. Measured at
 * 0.00181 log loss on the 630 warm-year predictions.
 *
 * This comment used to say the alternative "costs more folds than 16 seasons
 * can spare". That was wrong: scripts/model_baseline.js now refits the
 * calibration per year on strictly earlier residuals only, shrunk toward a = 1
 * with weight n/(n + 63), which needs no extra folds at all -- the walk-forward
 * predictions already form a time-ordered sequence to calibrate along. The
 * frozen baseline reports that as the headline. This function is unchanged and
 * still fits on what it is given; the discipline lives in the caller.
 */

const PROB_CLIP = 1e-3;

function clipProb(p) {
  return Math.min(1 - PROB_CLIP, Math.max(PROB_CLIP, p));
}

/* Log-gamma, Lanczos g=7. Needed only by the incomplete beta below. */
function logGamma(x) {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/* Continued fraction for the incomplete beta (Numerical Recipes betacf). */
function betacf(a, b, x) {
  const MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/* Regularised incomplete beta I_x(a,b). */
function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
  );
  return x < (a + 1) / (a + b + 2)
    ? bt * betacf(a, b, x) / a
    : 1 - bt * betacf(b, a, 1 - x) / b;
}

/* Student-t CDF. nu = Infinity is the normal, exactly. */
function studentTCdf(t, nu) {
  if (!(nu < 1e6)) return normalCdf(t);
  if (!isFinite(t)) return t > 0 ? 1 : 0;
  const p = 0.5 * betai(nu / 2, 0.5, nu / (nu + t * t));
  return t > 0 ? 1 - p : p;
}

/* Mean log loss of a calibration (a, nu) over walk-forward rows.
 * rows: [{m, p, sigma}] -- true margin, predicted margin, that fold's sigma. */
function logLossFor(rows, a, nu) {
  let s = 0;
  for (const r of rows) {
    if (r.m === 0) continue;              // a tie has no winner to score
    const p = clipProb(studentTCdf(a * r.p / r.sigma, nu));
    s += r.m > 0 ? -Math.log(p) : -Math.log(1 - p);
  }
  return s / rows.length;
}

/* Fit the link's scale and tail weight by minimising log loss.
 *
 * Two parameters over a coarse nu grid with a golden-section search on `a`
 * inside each. nu is searched on a grid rather than continuously because log
 * loss is very flat in it -- the data can tell 3 from 30, not 8 from 9 -- and a
 * grid keeps this cheap enough to re-run on every variable toggle. */
function calibrate(rows) {
  const NUS = [2, 3, 4, 6, 8, 12, 20, 40, Infinity];
  const GR = (Math.sqrt(5) - 1) / 2;
  let best = { a: 1, nu: Infinity, logLoss: Infinity };

  for (const nu of NUS) {
    let lo = 0.2, hi = 3.0;
    let c = hi - GR * (hi - lo), d = lo + GR * (hi - lo);
    let fc = logLossFor(rows, c, nu), fd = logLossFor(rows, d, nu);
    for (let i = 0; i < 30 && hi - lo > 1e-3; i++) {
      if (fc < fd) { hi = d; d = c; fd = fc; c = hi - GR * (hi - lo); fc = logLossFor(rows, c, nu); }
      else { lo = c; c = d; fc = fd; d = lo + GR * (hi - lo); fd = logLossFor(rows, d, nu); }
    }
    const a = (lo + hi) / 2;
    const ll = logLossFor(rows, a, nu);
    if (ll < best.logLoss) best = { a, nu, logLoss: ll };
  }
  return best;
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

/* Pairwise Pearson correlation between the enabled variables' standardised
 * differentials, over the same walk-forward training rows fitLinear() uses.
 *
 * WHY THIS IS A DIFFERENT CHECK FROM stability(). A coefficient can be
 * perfectly sign-stable across every held-out fold and still not mean what
 * it looks like: on the canonical set, Overall rating and National rank
 * correlate at ~0.99 and consistently draw a large coefficient of one sign
 * on one and the opposite sign on the other, every fold. stability() cannot
 * see that -- it only asks whether ONE coefficient agrees with itself across
 * folds, and both of these do. What is actually happening is that the two
 * columns carry almost the same information, so the fit is free to draw any
 * split between them that sums to the right net effect; the individual
 * numbers are an artefact of that split, not two independent effects that
 * happen to cancel. Only a correlation between the COLUMNS themselves
 * reveals that, which is what this computes.
 *
 * Returns a symmetric k x k matrix (k = cols.length), 1 on the diagonal, or
 * null if there are no training rows to measure it from.
 */
function pairwiseCorrelations(rows, cols, asOf) {
  const used = trainingRows(rows, asOf);
  const k = cols.length;
  const n = used.length;
  if (!n || !k) return null;

  const mean = new Array(k).fill(0);
  for (const r of used) for (let a = 0; a < k; a++) mean[a] += r.x[cols[a]];
  for (let a = 0; a < k; a++) mean[a] /= n;

  const cov = Array.from({ length: k }, () => new Array(k).fill(0));
  const varr = new Array(k).fill(0);
  for (const r of used) {
    const d = cols.map((c, a) => r.x[c] - mean[a]);
    for (let a = 0; a < k; a++) {
      varr[a] += d[a] * d[a];
      for (let b = a; b < k; b++) cov[a][b] += d[a] * d[b];
    }
  }

  const corr = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let a = 0; a < k; a++) {
    for (let b = a; b < k; b++) {
      const denom = Math.sqrt(varr[a] * varr[b]);
      // A variable with zero variance in this sample cannot correlate with
      // anything; 0 rather than NaN so a caller need not special-case it.
      const c = denom > 1e-9 ? cov[a][b] / denom : (a === b ? 1 : 0);
      corr[a][b] = c;
      corr[b][a] = c;
    }
  }
  return corr;
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
    // sigma travels with the row: each fold has its own, and the calibration
    // below is fitted across folds, so the two cannot be collapsed.
    for (const r of test) {
      pooled.push({ x: r.x, m: r.m, p: predictMargin(f.beta, cols, r.x), sigma: f.sigma });
    }
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

  // Calibrate the link on these held-out rows, and report what it bought
  // against the old uncalibrated normal so the change is auditable rather
  // than asserted. Both numbers are over the same games.
  const calibration = calibrate(pooled);
  const before = { a: 1, nu: Infinity };
  const scoreProb = (cal) => {
    let ll = 0, brier = 0, m = 0, pinned = 0;
    for (const r of pooled) {
      if (r.m === 0) continue;
      const p = clipProb(studentTCdf(cal.a * r.p / r.sigma, cal.nu));
      const y = r.m > 0 ? 1 : 0;
      ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      brier += (p - y) ** 2;
      if (p >= 1 - 1e-4 || p <= 1e-4) pinned++;
      m++;
    }
    return m ? { logLoss: ll / m, brier: brier / m, pinned, n: m } : null;
  };

  return {
    n,
    seasons: folds,
    rmse: Math.sqrt(sse / n),
    mae: sae / n,
    r2: sst > 0 ? 1 - sse / sst : null,
    accuracy: decided ? correct / decided : null,
    sigma: sigmaSum / folds,
    calibration,
    probScore: scoreProb(calibration),
    probScoreUncalibrated: scoreProb(before),
    perYear,
    stability: stability(trajectory),
    // Kept so a caller can re-score these same held-out games under a
    // DIFFERENT link than the one fitted here -- causalWalkForward() shrinks
    // `a` before use, and the reliability table has to be built with the
    // link the board actually applies, not the unshrunk one.
    pooled,
  };
}

/* Reliability table: when the model says 70%, how often does it happen?
 *
 * This is the one check the accuracy figure cannot stand in for. Accuracy
 * grades the PICK (was the favourite right); it is unchanged by any monotone
 * reparameterisation of the link, so a model whose 70%s come true 90% of the
 * time and one whose 70%s come true 55% of the time can have identical
 * accuracy. The board prints the percentage, so the percentage is a separate
 * claim and needs its own evidence -- until now the page asserted "when it
 * says 70% it is right about 70%" with nothing behind it (2026-09 review).
 *
 * ORIENTED TO THE FAVOURITE. Every game is a pair (p, 1-p); binning the raw
 * team-A probability would put the same game in two bins depending on which
 * team the row happened to list first. Flipping to p >= 0.5 makes each game
 * count once, in the band the model's confidence actually sits in.
 *
 * `se` is the binomial standard error UNDER THE MODEL'S OWN CLAIM -- sqrt(p(1-p)/n)
 * at the bin's mean predicted p, not at the observed rate. That is the right
 * null: "if these percentages were honest, the observed rate would land
 * within about this much of them". At the observed rate a bin that went
 * 30-for-30 would report zero uncertainty, which is not what 30 games tells
 * you.
 *
 * READ THE BIN EDGES WITH THE WARNING IN calibrate()'s comment above: the
 * 0.5-0.6 bin is sensitive to where its lower edge falls, because a handful
 * of coin-flip games crossing an arbitrary line is a large fraction of a
 * small bin. The table is evidence about the band as a whole, not a verdict
 * on any one row of it.
 *
 * rows : crossValidate()'s `pooled` -- {m, p (predicted MARGIN), sigma}
 * cal  : {a, nu}, the link to score them under
 */
const RELIABILITY_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9];

function reliabilityTable(rows, cal, edges) {
  const e = edges || RELIABILITY_EDGES;
  const bins = e.slice(0, -1).map((lo, i) => ({ lo, hi: e[i + 1], n: 0, sumP: 0, wins: 0 }));
  for (const r of rows) {
    if (r.m === 0) continue;                       // a tie has no winner to score
    let p = clipProb(studentTCdf(cal.a * r.p / Math.max(r.sigma, 1e-6), cal.nu));
    let y = r.m > 0 ? 1 : 0;
    if (p < 0.5) { p = 1 - p; y = 1 - y; }
    const b = bins.find(b => p >= b.lo && p < b.hi);
    if (!b) continue;
    b.n++; b.sumP += p; b.wins += y;
  }
  return bins.map(b => {
    const predicted = b.n ? b.sumP / b.n : null;
    return {
      lo: b.lo, hi: Math.min(b.hi, 1),
      n: b.n,
      predicted,
      actual: b.n ? b.wins / b.n : null,
      se: b.n ? Math.sqrt(predicted * (1 - predicted) / b.n) : null,
    };
  });
}

/* Walk-forward evaluation and calibration for a DISPLAYED season, causally.
 *
 * crossValidate() above fits the link's (a, nu) on every held-out row it is
 * handed. Handing it the whole matrix and then using that calibration to show
 * season Y means two link parameters were estimated from Y's own outcomes --
 * and from every later season's. The margins were out of sample; the
 * calibration was not. Found in the 2026-09 methodology audit (Step 2, P1-1):
 * the page used one global a=1.53, nu=3 for every season from 2014 to 2026.
 *
 * This is the caller-side discipline scripts/model_baseline.js and
 * src/prediction/pit_production_model.py already apply, moved into fit.js so
 * the page cannot drift from them again: only seasons strictly before `asOf`
 * are evaluated, and `a` is shrunk toward 1 with weight n / (n + 63) because
 * the per-year fits are noisy (one season is 63 binary outcomes). For the
 * prospective season nothing is lost -- every prior row is still used -- and
 * for a historical season the displayed probabilities no longer know how that
 * tournament ended.
 */
const CAL_PRIOR_STRENGTH = 63;

function causalWalkForward(rows, cols, years, asOf, minYear) {
  const priorRows = rows.filter(r => r.y < asOf);
  const priorYears = years.filter(y => y < asOf);
  const oos = crossValidate(priorRows, cols, priorYears, minYear);
  if (!oos) return null;
  const n = oos.n;
  const w = n / (n + CAL_PRIOR_STRENGTH);
  oos.calibrationRaw = oos.calibration;
  oos.calibration = {
    a: w * oos.calibration.a + (1 - w) * 1,
    nu: oos.calibration.nu,
    priorN: n,
    shrinkWeight: w,
  };
  // Under the link the board will actually use -- the shrunk one -- and over
  // only the seasons strictly before asOf, same as everything else here.
  oos.reliability = reliabilityTable(oos.pooled, oos.calibration);
  return oos;
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

/* P(team1 wins), from a predicted margin and the fit's residual spread.
 *
 * `cal` is the {a, nu} fitted by calibrate() on held-out games. Omitting it
 * falls back to the raw normal link (a = 1, nu = Infinity), which is the
 * pre-calibration behaviour -- kept as the default so the fallback is the
 * conservative one when no walk-forward evaluation was possible. The clip
 * applies either way. */
/* k-nearest-neighbour margin prediction over past tournament games.
 *
 * WHAT IT IS. The query is a matchup's standardised differential on whichever
 * variables are switched on. Its k closest historical matchups by Euclidean
 * distance vote, and their mean margin is the prediction. Where the ridge fit
 * asks "what does the average game say about these variables", this asks "what
 * happened in the games that looked most like this one".
 *
 * WALK-FORWARD, same rule as fitLinear. Only rows from seasons strictly before
 * asOf are eligible, so the season on screen never votes on itself.
 *
 * ANTISYMMETRY IS BUILT IN BY SEARCHING BOTH ORIENTATIONS. Every row is
 * considered as itself and as its mirror (-x, -m). The neighbour set for a
 * query is therefore the exact mirror of the set for the swapped query, and
 * their mean margins negate. Without this, kNN would break the property the
 * board depends on: nothing forces the neighbours of x to be the mirrors of
 * the neighbours of -x when only one orientation is stored.
 *
 * SIGMA IS LOCAL. The spread of the neighbours' margins is a better
 * uncertainty estimate here than a global residual would be -- a query sitting
 * among tightly-agreeing games genuinely is more certain than one among
 * scattered ones. Floored so an unlucky set of identical neighbours cannot
 * produce infinite confidence.
 *
 * The calibration passed to the link was fitted for the ridge model, so the
 * probabilities this produces are approximate in a way the ridge board's are
 * not. It is an exploration surface, not the frozen baseline.
 */
function knnPredict(rows, cols, x, k, asOf) {
  const pool = asOf === null || asOf === undefined ? rows : rows.filter(r => r.y < asOf);
  if (!pool.length || !cols.length) return null;

  const cand = [];
  for (const r of pool) {
    let d = 0;
    for (let i = 0; i < cols.length; i++) {
      const diff = x[i] - r.x[cols[i]];
      d += diff * diff;
    }
    cand.push({ d, m: r.m });          // as stored
    let dm = 0;
    for (let i = 0; i < cols.length; i++) {
      const diff = x[i] + r.x[cols[i]];
      dm += diff * diff;
    }
    cand.push({ d: dm, m: -r.m });     // mirrored
  }

  const kk = Math.max(1, Math.min(k, cand.length));
  cand.sort((p, q) => p.d - q.d);
  const near = cand.slice(0, kk);

  let mean = 0;
  for (const c of near) mean += c.m;
  mean /= kk;
  let v = 0;
  for (const c of near) v += (c.m - mean) ** 2;
  const sigma = Math.max(Math.sqrt(v / Math.max(kk - 1, 1)), 1e-6);
  return { margin: mean, sigma, n: kk, pool: pool.length };
}

function winProbFromMargin(margin, sigma, cal) {
  const a = cal && isFinite(cal.a) ? cal.a : 1;
  const nu = cal ? cal.nu : Infinity;
  return clipProb(studentTCdf(a * margin / Math.max(sigma, 1e-6), nu));
}

/* Each team's probability of reaching every round of a real bracket, from
 * pairwise game probabilities alone.
 *
 * WHY THIS IS NOT JUST winProb() REPEATED. A team's chance of reaching the
 * Sweet 16 is not its chance of winning one more game -- it depends on WHO
 * shows up in that game, and that opponent is themselves uncertain: they
 * still have to win their own earlier game to be there at all. The correct
 * quantity marginalises over that: P(t survives r rounds) = P(t survived
 * r-1) * sum over every possible round-r opponent o of P(o survives to meet
 * t) * P(t beats o). Skipping the opponent's own survival term and just
 * multiplying win probabilities down one column would silently assume every
 * possible opponent is equally certain to arrive, which overstates the
 * favourite's odds in every later round.
 *
 * THE MERGE IS BOTTOM-UP OVER THE REAL TREE, not a seed-based table. Two
 * sibling subtrees of size 2^r each already carry a full round-by-round
 * survival distribution for their own teams; merging them for round r+1
 * needs only those distributions and the pairwise win function, so the exact
 * matchups the real bracket produces are respected without simulating a
 * single tournament.
 *
 * order   team ids/indices in real bracket order (the same order first_round
 *         uses), length a power of two.
 * winProb(a, b) -> P(a beats b), assumed antisymmetric: winProb(a,b) ===
 *         1 - winProb(b,a). Nothing here enforces that; a caller whose
 *         estimator does not have it will get a result that likewise does not
 *         sum to 1 per round, which is the same property test that catches a
 *         broken caller in tests/test_advancement.js.
 *
 * Returns { [team]: number[] }, one entry per team, each an array indexed by
 * round: index 0 is "won the Round of 64 game" (reached the Round of 32),
 * index 5 (for 64 teams) is "won the championship game" -- the same round
 * numbering solveByFit() uses for its six-round board, so a caller can read
 * probs[team][r] straight off ROUNDS[r].
 *
 * MONOTONE BY CONSTRUCTION, NOT BY POLICY: each round's probability is the
 * previous round's probability times a factor in [0, 1] (a weighted average
 * of win probabilities), so it can only fall or hold round over round, never
 * rise. A caller seeing an increase has a bug in winProb, not in this
 * function.
 */
function bracketAdvancementProbs(order, winProb) {
  const n = order.length;
  const rounds = Math.log2(n);
  if (!Number.isInteger(rounds) || rounds < 1) {
    throw new Error(`bracketAdvancementProbs: order length must be a power of two >= 2, got ${n}`);
  }

  let nodes = order.map(t => ({ teams: [t], probs: { [t]: [] } }));
  for (let r = 0; r < rounds; r++) {
    const next = [];
    for (let i = 0; i < nodes.length; i += 2) {
      const L = nodes[i], R = nodes[i + 1];
      const priorL = t => (r === 0 ? 1 : L.probs[t][r - 1]);
      const priorR = t => (r === 0 ? 1 : R.probs[t][r - 1]);

      const probs = {};
      for (const t of L.teams) probs[t] = L.probs[t].slice();
      for (const t of R.teams) probs[t] = R.probs[t].slice();

      for (const t of L.teams) {
        let winThisRound = 0;
        for (const o of R.teams) winThisRound += priorR(o) * winProb(t, o);
        probs[t].push(priorL(t) * winThisRound);
      }
      for (const t of R.teams) {
        let winThisRound = 0;
        for (const o of L.teams) winThisRound += priorL(o) * winProb(t, o);
        probs[t].push(priorR(t) * winThisRound);
      }
      next.push({ teams: [...L.teams, ...R.teams], probs });
    }
    nodes = next;
  }
  return nodes[0].probs;
}

/* ---------------------------------------------------------------- risk-aware scoring & greedy fill
 *
 * JS port of `_make_ev_scorer` / `_decide_winner` / `_walk_bracket` (forward_greedy
 * mode) in src/optimization/bracket_construction.py, for the one strategy that has
 * no server-side risk grid: the fitted model. p1 and ev already ship a precomputed
 * five-point `risk_variants` grid built by that same Python formula; the fitted
 * board cannot be precomputed because it recomputes reactively as the user toggles
 * variables, so this is the identical arithmetic, run here instead.
 *
 * FORWARD_GREEDY, NOT region_top_n's BEAM SEARCH. construct_bracket's modes all
 * share this scorer and `_decide_winner`'s argmax-with-tiebreak; they differ only
 * in which games get an anchor lock before the forward walk runs. region_top_n
 * additionally replaces each region's forward walk with a beam search over whole-
 * region EV (~2^15 outcomes per region) -- a global per-region optimisation, not a
 * per-game decision, and re-running it here on every filter toggle would be a much
 * larger reactive cost for a difference build_ui_payload.py's own ev_optimal comment
 * describes as "the same bracket to within one game" for the EV case. forward_greedy
 * has no anchor at all: every game, R64 through CHAMP, is decided by this scorer's
 * argmax, with the same lexical tie-break, and nothing else. That is what is ported
 * here: exactly `construct_bracket(mode="forward_greedy")`'s walk, which is directly
 * checkable against the real Python function.
 *
 * ROUND_PROBS SHAPE MATCHES bracketAdvancementProbs()'s OUTPUT: {team: number[6]},
 * index 0 = P(win the R64 game) ... index 5 = P(win the CHAMP game) -- so
 * state.advancement in app.js is exactly this function's `roundProbs` input,
 * already computed live from the fitted model. Its keys are whatever
 * `first_round` (state.season.first_round) is keyed by -- in the current
 * payload, integer indices into `teams`, not the string team_id.
 *
 * PUBLIC_PICKS SHAPE MATCHES THE SEASON PAYLOAD'S: {team_id: {round_name: pick_rate}},
 * the same object build_ui_payload.py ships as the season's top-level
 * `public_picks` field (real archived ESPN picks, the same source _make_ev_scorer's
 * Python callers use). It is keyed by the STRING team_id, not the bracket-position
 * index round_probs and first_round use -- see `teams` below for why every function
 * here also takes the season's `teams` array: it is what resolves one key space
 * into the other, exactly as build_ui_payload.py's own `seeds = {t["id"]: ...}`
 * and this port's Python counterpart key everything by team_id.
 */

const RISK_ROUND_NAMES = ['R64', 'R32', 'S16', 'E8', 'F4', 'CHAMP'];

/* ESPN standard scoring -- mirrors `_DEFAULT_SCORING` in bracket_construction.py. */
const DEFAULT_SCORING = { R64: 10, R32: 20, S16: 40, E8: 80, F4: 160, CHAMP: 320 };

/* Mirrors `POOL_FACTOR_THRESHOLD` in bracket_construction.py: at or below this
 * entry count, in the default "threshold" mode, pool size does not affect the
 * score at all. */
const POOL_FACTOR_THRESHOLD = 50;

/* Build a (teamId, roundIdx) -> score closure, the same formula as
 * `_make_ev_scorer` in src/optimization/bracket_construction.py:
 *
 *   score = model_prob * pts * blended_diff
 *   blended_diff = (1 - risk_level) * 1.0 + risk_level * diff
 *   diff = uniqueness * pool_factor
 *   uniqueness = max(0, 1 - public_prob)
 *   pool_factor = 1 / max(1, log2(max(pool_size * public_prob, 1)))   (see modes below)
 *
 * roundProbs    {team: number[6]}            -- e.g. state.advancement, keyed
 *               the same way `firstRound` is (bracket-position index)
 * publicPicks   {team_id: {round_name: number}} -- the season payload's
 *               public_picks, keyed by the STRING team_id
 * teams         state.season.teams -- teams[index].id resolves a roundProbs/
 *               firstRound key to the team_id publicPicks is keyed by
 * riskLevel     0 (chalk) .. 1 (contrarian)
 * poolSize      entries in the pool
 * scoringSystem {round_name: points}, default DEFAULT_SCORING
 * options.confidenceThreshold  same "lock high-certainty games to chalk" escape
 *   hatch as the Python function; null/omitted disables it (this port's default,
 *   and _make_ev_scorer's).
 * options.poolFactorMode  "threshold" (default, matches every shipped bracket) |
 *   "continuous" | "off" -- see _make_ev_scorer's docstring for what each does.
 */
function makeRiskScorer(roundProbs, publicPicks, teams, riskLevel, poolSize, scoringSystem, options) {
  const opts = options || {};
  const confidenceThreshold = opts.confidenceThreshold == null ? null : opts.confidenceThreshold;
  const poolFactorMode = opts.poolFactorMode || 'threshold';
  const scoring = scoringSystem || DEFAULT_SCORING;
  if (poolFactorMode !== 'threshold' && poolFactorMode !== 'continuous' && poolFactorMode !== 'off') {
    throw new Error(`makeRiskScorer: poolFactorMode must be 'threshold', 'continuous' or 'off', got ${poolFactorMode}`);
  }

  return function score(teamId, roundIdx) {
    const roundName = RISK_ROUND_NAMES[roundIdx];
    const rp = roundProbs[teamId];
    const modelProb = rp && rp[roundIdx] != null ? Number(rp[roundIdx]) || 0 : 0;
    const tid = teams && teams[teamId] ? teams[teamId].id : teamId;
    const pp = publicPicks[tid];
    const publicProb = pp && pp[roundName] != null ? Number(pp[roundName]) || 0 : 0;
    const pts = scoring[roundName] != null ? Number(scoring[roundName]) : 10;

    // Confidence routing: lock high-certainty games to chalk so the risk_level
    // budget is concentrated on genuinely uncertain matchups.
    if (confidenceThreshold !== null && modelProb >= confidenceThreshold) {
      return modelProb * pts;
    }

    const uniqueness = Math.max(0, 1 - publicProb);
    let poolFactor = 1.0;
    if (poolFactorMode === 'threshold') {
      if (poolSize > POOL_FACTOR_THRESHOLD) {
        const expectedDups = poolSize * publicProb;
        poolFactor = 1 / Math.max(1, Math.log2(Math.max(expectedDups, 1)));
      }
    } else if (poolFactorMode === 'continuous') {
      const expectedDups = poolSize * publicProb;
      poolFactor = 1 / Math.max(1, Math.log2(Math.max(expectedDups, 1)));
    }
    const diff = uniqueness * poolFactor;

    const blendedDiff = (1 - riskLevel) * 1.0 + riskLevel * diff;
    return modelProb * pts * blendedDiff;
  };
}

/* One game's winner: argmax score(), lexical tie-break on the team_id STRING --
 * mirrors the priority-3 fallback in `_decide_winner` exactly (the only priority
 * that applies here: this port has no forced-champion or anchor-lock concept,
 * matching construct_bracket's "forward_greedy" mode called with no
 * forced_champion). `teams` resolves each bracket-position index to the team_id
 * `_decide_winner` compares -- comparing the indices themselves would only agree
 * with Python's tie-break by coincidence of the payload's team ordering. */
function decideRiskWinner(t1, t2, roundIdx, scorer, teams) {
  const s1 = scorer(t1, roundIdx);
  const s2 = scorer(t2, roundIdx);
  if (Math.abs(s1 - s2) > 1e-9) return s1 > s2 ? t1 : t2;
  const id1 = teams && teams[t1] ? teams[t1].id : t1;
  const id2 = teams && teams[t2] ? teams[t2].id : t2;
  return id1 < id2 ? t1 : t2;
}

/* Forward-greedy fill of the whole bracket, R64 through CHAMP -- mirrors
 * `_walk_bracket` with locked_teams=set(), lock_through_round=null, no
 * forced_champion (construct_bracket's "forward_greedy" mode). No explicit
 * region/seed bookkeeping is needed the way the Python version has it: the
 * caller's `firstRound` is already the season's real bracket order (the same
 * flat 64-team list decodeBracket() and solveBracket() in app.js walk), so
 * pairing adjacent teams round over round reproduces the same region-by-region,
 * round-by-round structure by construction.
 *
 * `teams` is state.season.teams, passed through to decideRiskWinner() for its
 * team_id tie-break -- see makeRiskScorer()'s comment for why every function
 * here needs it.
 *
 * Returns picks in decodeBracket()'s shape: an array of 6 round-lists of
 * winners (32, 16, 8, 4, 2, 1 teams).
 */
function riskGreedyBracket(firstRound, scorer, teams) {
  let current = firstRound.slice();
  const rounds = [];
  for (let r = 0; r < 6; r++) {
    const next = [];
    for (let g = 0; g < current.length; g += 2) {
      next.push(decideRiskWinner(current[g], current[g + 1], r, scorer, teams));
    }
    rounds.push(next);
    current = next;
  }
  return rounds;
}

/* Expected points and variance for a complete 6-round picks array, under the
 * independent-pick approximation -- mirrors `_compute_expected_points`. `picks`
 * is riskGreedyBracket()'s (or decodeBracket()'s) shape; `roundProbs` is the same
 * {team: number[6]} the scorer above took. */
function riskExpectedPoints(picks, roundProbs, scoringSystem) {
  const scoring = scoringSystem || DEFAULT_SCORING;
  let ev = 0, variance = 0;
  for (let r = 0; r < picks.length; r++) {
    const pts = scoring[RISK_ROUND_NAMES[r]] != null ? Number(scoring[RISK_ROUND_NAMES[r]]) : 0;
    for (const team of picks[r]) {
      const rp = roundProbs[team];
      const p = rp && rp[r] != null ? Number(rp[r]) || 0 : 0;
      ev += p * pts;
      variance += p * (1 - p) * pts * pts;
    }
  }
  return { ev, variance };
}

/* The whole job: fitted-model round probabilities + real public picks + a risk
 * level -> a complete greedy-filled bracket, in the shape the rest of the page
 * expects. Convenience wrapper around makeRiskScorer + riskGreedyBracket so a
 * caller (app.js's slider wiring) does not have to build either closure itself.
 * `teams` is state.season.teams -- see makeRiskScorer()'s comment. */
function solveByRisk(firstRound, roundProbs, publicPicks, teams, riskLevel, poolSize, scoringSystem, options) {
  const scorer = makeRiskScorer(roundProbs, publicPicks, teams, riskLevel, poolSize, scoringSystem, options);
  return riskGreedyBracket(firstRound, scorer, teams);
}

/* What one variable predicts on its own, walk-forward.
 *
 * Two questions, both answered only from rows strictly before `asOf` -- the
 * same window the model trains on, so nothing here has seen the season on
 * screen:
 *
 *   betterWins  How often did the team with the better value win? x is the
 *               standardised differential, already sign-corrected so x > 0
 *               means team1 is better on this variable. Rows with x == 0
 *               (no edge) are excluded; rows with m == 0 cannot exist (the
 *               matrix asserts it). Reported overall and by round, each with
 *               a binomial standard error, because the E8/F4/NCG cells are
 *               small (4/2/1 games a season) and a rate without its error
 *               invites reading noise as a trend.
 *   corr        Pearson correlation between the differential and the margin,
 *               the "relationship with tournament game margin".
 *
 * This is deliberately NOT a fit: it is the raw evidence a variable brings,
 * before any other variable is allowed to explain it away. The single-variable
 * model (fitLinear on one column, walk-forward) is the fitted counterpart and
 * lives beside this in the page.
 */
function variableRecord(rows, col, asOf) {
  const used = trainingRows(rows, asOf);
  const tally = { all: { n: 0, w: 0 } };
  let sx = 0, sm = 0, n = 0;
  for (const r of used) {
    const x = r.x[col];
    if (x === 0 || r.m === 0) continue;
    n++; sx += x; sm += r.m;
    const win = (x > 0) === (r.m > 0);
    tally.all.n++; tally.all.w += win ? 1 : 0;
    if (r.r) {
      (tally[r.r] ||= { n: 0, w: 0 });
      tally[r.r].n++; tally[r.r].w += win ? 1 : 0;
    }
  }
  if (!n) return null;
  const mx = sx / n, mm = sm / n;
  let cov = 0, vx = 0, vm = 0;
  for (const r of used) {
    const x = r.x[col];
    if (x === 0 || r.m === 0) continue;
    cov += (x - mx) * (r.m - mm); vx += (x - mx) ** 2; vm += (r.m - mm) ** 2;
  }
  const rate = t => {
    const p = t.w / t.n;
    return { n: t.n, rate: p, se: Math.sqrt(p * (1 - p) / t.n) };
  };
  const byRound = {};
  for (const k of Object.keys(tally)) if (k !== 'all') byRound[k] = rate(tally[k]);
  return { n, corr: vx > 0 && vm > 0 ? cov / Math.sqrt(vx * vm) : 0, betterWins: rate(tally.all), byRound };
}

/* Model sensitivity: the model refit WITHOUT one variable.
 *
 * Preregistered in artifacts/methodology_audit/ui_phase_b/
 * PREREGISTRATION_MODEL_SENSITIVITY.md; this function is that definition.
 *
 *   - `cols` minus one column, refit by fitLinear on the same rows and the
 *     same asOf boundary. Every remaining coefficient is re-estimated: the
 *     excluded one is not zeroed, and the others are not copied over.
 *   - The link is calibrated by the same causalWalkForward on the exclusion
 *     model's own held-out predictions, same folds, same shrinkage.
 *   - The held-out accuracy and log loss come back beside the column index
 *     so a caller can put them next to the full model's on the same games.
 *
 * WHAT THE NUMBER IS NOT. With correlated columns, removing one lets the
 * others absorb it, so a small change is not "this information is
 * unimportant" and a large change is not "this variable is a cause". It is
 * exactly what it says: the model without this column, refit.
 *
 * Returns an array parallel to `cols`: [{ col, cols: remaining, fit, oos }],
 * where `fit` is fitLinear's result (beta over `remaining`) and `oos` is
 * causalWalkForward's (null if too little history). Entries whose fit is
 * not ok are still returned, with fit.ok false, so a caller can say so.
 */
function exclusionModels(rows, cols, years, asOf, minYear) {
  return cols.map((c, j) => {
    const remaining = cols.filter((_, k) => k !== j);
    const fit = fitLinear(rows, remaining, asOf);
    const oos = fit.ok ? causalWalkForward(rows, remaining, years, asOf, minYear) : null;
    return { col: c, cols: remaining, fit, oos };
  });
}

/* ---------------------------------------------------------------- rule search
 *
 * EXPERIMENTAL, FOUND AFTER THE FACT, NOT A MODEL. A "rule" is one criterion
 * per round: in that round every game goes to the team with the better
 * value of one variable (direction-corrected z, higher is better; ties to the
 * better seed, then the lower index -- the board's own tie rule). The
 * question the search answers is HOW FAR BACK one rule can reproduce the
 * chosen rounds: starting from the newest played season, the window grows
 * one season back while any rule still reproduces every season in it
 * (ruleRun). What survives is a description of those seasons, not evidence
 * about the next one. Measured on the shipped data (Final Four + champion,
 * 32 criteria) for the 2027 page: a rule of at most two variables reaches
 * one season (2026; 72 rules); of at most three, two seasons (2025-2026;
 * 39 rules), and none reaches 2024. Without a cap the survivors run to
 * millions (5.25M for 2025 alone, ~600 MB) with no longer run, so the cap
 * is applied inside the enumeration and no uncapped search is offered.
 *
 * season: { year, first_round: number[64], crit: {key: number[64]},
 *           seed: number[64], actual: number[][] (winners per round) }
 * keys:   the criterion keys, in a fixed order shared by all seasons
 * checkpoints: round indices whose WINNERS must match `actual`
 *           (1 = Sweet 16, 2 = Elite Eight, 3 = Final Four, 5 = champion)
 *
 * Sequences are encoded as integers base keys.length, most significant digit
 * = round 0, so sets of them intersect cheaply across seasons.
 */
function rulePlay(field, crit, seed, k) {
  const out = new Array(field.length / 2);
  for (let g = 0; g < field.length; g += 2) {
    const a = field[g], b = field[g + 1];
    const va = crit[k][a], vb = crit[k][b];
    let w;
    if (va !== vb) w = va > vb ? a : b;
    else if (seed[a] !== seed[b]) w = seed[a] < seed[b] ? a : b;
    else w = Math.min(a, b);
    out[g / 2] = w;
  }
  return out;
}

function sameSet(arr, target) {
  if (arr.length !== target.size) return false;
  for (const t of arr) if (!target.has(t)) return false;
  return true;
}

/* Set bits of a criterion mask: how many distinct criteria a sequence uses. */
function bitCount(m) {
  let n = 0;
  for (; m; m &= m - 1) n++;
  return n;
}

/* Every criterion sequence over rounds 0..lastRound that reproduces the
 * season's checkpoints, using at most `maxCriteria` distinct criteria (null:
 * any). Depth-first over rounds with the surviving field as the memo key,
 * so the work is proportional to distinct states, not to keys^rounds. The
 * cap is applied as sequences are extended -- each carries the mask of the
 * criteria it has used -- so a capped search never builds the millions of
 * uncapped survivors; the set is identical to filtering afterwards. Returns
 * a Set of encoded sequences. */
function ruleSequencesForSeason(season, keys, checkpoints, lastRound, maxCriteria = null) {
  const B = keys.length;
  const targets = {};
  for (const r of checkpoints) targets[r] = new Set(season.actual[r]);
  let level = new Map([[season.first_round.join(','), { field: season.first_round.slice(), seqs: [0], masks: [0] }]]);
  for (let r = 0; r <= lastRound; r++) {
    const next = new Map();
    for (const { field, seqs, masks } of level.values()) {
      for (let ki = 0; ki < B; ki++) {
        const nf = rulePlay(field, season.crit, season.seed, keys[ki]);
        if (targets[r] && !sameSet(nf, targets[r])) continue;
        const id = nf.join(',');
        const bit = 1 << ki;
        let e = next.get(id);
        for (let i = 0; i < seqs.length; i++) {
          const m = masks[i] | bit;
          if (maxCriteria !== null && bitCount(m) > maxCriteria) continue;
          if (!e) { e = { field: nf, seqs: [], masks: [] }; next.set(id, e); }
          e.seqs.push(seqs[i] * B + ki);
          e.masks.push(m);
        }
      }
    }
    level = next;
  }
  const out = new Set();
  for (const { seqs } of level.values()) for (const sq of seqs) out.add(sq);
  return out;
}

function decodeRule(code, keys, nRounds) {
  const B = keys.length, out = new Array(nRounds);
  for (let i = nRounds - 1; i >= 0; i--) { out[i] = keys[code % B]; code = Math.floor(code / B); }
  return out;
}

/* (distinct criteria, switches): the complexity the results are ranked by. */
function ruleComplexity(seq) {
  let switches = 0;
  for (let i = 1; i < seq.length; i++) if (seq[i] !== seq[i - 1]) switches++;
  return [new Set(seq).size, switches];
}

/* The same two numbers read off an encoded sequence, without decoding it:
 * a season can leave two million survivors, and a comparator that decoded
 * and re-counted each one on every comparison was 15 s of a 0.3 s search. */
function ruleComplexityOfCode(code, B, nRounds) {
  let switches = 0, prev = -1, mask = 0;
  for (let i = nRounds - 1; i >= 0; i--) {
    const d = code % B; code = Math.floor(code / B);
    mask |= 1 << d;
    if (i < nRounds - 1 && d !== prev) switches++;
    prev = d;
  }
  return [bitCount(mask), switches];
}

/* Composite ratings and rank-of-ratings: a team's whole season folded into
 * one number (or, for seed, the committee's own). Real, and correlated with
 * winning, but the opposite of what the rule search is meant to surface --
 * "this specific thing about a team predicts this specific round." Kept
 * out of ruleComplexityOfCode's tested two-number contract; counted
 * separately, once per survivor (never inside a sort comparator), by
 * ruleGeneralCount below. Tempo and the offense/defense splits stay out of
 * this set -- pace and a side of the ball are specific, not a strength
 * rating. */
const RULE_GENERAL_KEYS = new Set([
  'barthag', 't_rank', 'massey_avg_rank', 'sos_avg_opp_barthag', 'srs',
  'adj_offensive_efficiency', 'adj_defensive_efficiency', 'seed',
]);

/* How many distinct RULE_GENERAL_KEYS criteria (by `generalMask`, a bitmask
 * over the same key indices ruleComplexityOfCode uses) a sequence uses.
 * Decoded the same cheap way, as its own function so ruleComplexityOfCode's
 * two-element return (asserted against ruleComplexity() in tests) is
 * untouched. */
function ruleGeneralCount(code, B, nRounds, generalMask) {
  if (!generalMask) return 0;
  let mask = 0;
  for (let i = nRounds - 1; i >= 0; i--) { mask |= 1 << (code % B); code = Math.floor(code / B); }
  return bitCount(mask & generalMask);
}

/* How far back one rule can reproduce the checkpoints. `seasons` newest
 * first. Skip phase: a newest season that no rule reproduces on its own is
 * recorded in `skipped` and passed over -- the run has to start at the
 * most recent season some rule can reproduce. Run phase: from that season,
 * the window grows one season back while the intersection of survivors is
 * non-empty; `stoppedAt` is the season that left none (null if every
 * remaining season was consumed). Seasons past the break are never
 * enumerated. Survivors are ranked fewest RULE_GENERAL_KEYS criteria first
 * (a rule built from specific, non-composite variables is offered ahead of
 * an equally simple one that leans on a rating or a rank, when one
 * survives), then simplest: fewest distinct criteria, then fewest
 * switches, then by the encoded sequence -- criterion index per round,
 * round 0 most significant, which with `keys` sorted is alphabetical round
 * by round; one number per rule in a typed-array sort. `generalKeys`
 * (default RULE_GENERAL_KEYS; pass null/empty Set to turn the tie-break
 * off) never changes which sequences survive, only their order. */
function ruleRun(seasons, keys, checkpoints, maxCriteria, generalKeys = RULE_GENERAL_KEYS) {
  const cps = [...new Set(checkpoints)].sort((a, b) => a - b);
  if (!cps.length) throw new Error('ruleRun: no checkpoints');
  const lastRound = cps[cps.length - 1];
  const cpSet = new Set(cps);
  const B = keys.length, nRounds = lastRound + 1, M = Math.pow(B, nRounds);
  if (B > 32) throw new Error(`ruleRun: ${B} criteria; the complexity mask holds 32`);
  const generalMask = generalKeys ? keys.reduce((m, k, ki) => generalKeys.has(k) ? m | (1 << ki) : m, 0) : 0;
  const run = [], skipped = [];
  let S = null, stoppedAt = null;
  for (const sn of seasons) {
    const s = ruleSequencesForSeason(sn, keys, cpSet, lastRound, maxCriteria);
    if (S === null) {
      if (!s.size) { skipped.push(sn.year); continue; }
      S = s; run.push(sn.year); continue;
    }
    const next = new Set([...S].filter(c => s.has(c)));
    if (!next.size) { stoppedAt = sn.year; break; }
    S = next; run.push(sn.year);
  }
  if (S === null) return { rules: [], run, skipped, stoppedAt };
  // (general count * 33 + distinct criteria) leaves distinct (< 33) its own
  // decimal digit under general, exactly as (distinct * 8 + switches) already
  // leaves switches (< 8) its own digit under distinct -- same trick, one
  // more tier, still one comparable float per rule.
  const order = new Float64Array(S.size);
  let i = 0;
  for (const code of S) {
    const [distinct, switches] = ruleComplexityOfCode(code, B, nRounds);
    const general = ruleGeneralCount(code, B, nRounds, generalMask);
    order[i++] = ((general * 33 + distinct) * 8 + switches) * M + code;
  }
  order.sort();
  const rules = new Array(order.length);
  for (let j = 0; j < order.length; j++) rules[j] = decodeRule(order[j] % M, keys, nRounds);
  return { rules, run, skipped, stoppedAt };
}

/* Apply a rule to a season until one team remains (six rounds for 64). Rounds past the last
 * constrained one reuse the last criterion (stated on the page). Returns the
 * board shape solveBracket() produces: rounds of {a, b, win}. */
function ruleBracket(season, seq) {
  let field = season.first_round.slice();
  const rounds = [];
  for (let r = 0; field.length > 1; r++) {
    const k = seq[Math.min(r, seq.length - 1)];
    const next = rulePlay(field, season.crit, season.seed, k);
    const games = [];
    for (let g = 0; g < field.length; g += 2) games.push({ a: field[g], b: field[g + 1], win: next[g / 2] });
    rounds.push(games);
    field = next;
  }
  return rounds;
}

/* Does the rule reproduce the checkpoints in this season? For the honest
 * check on seasons OUTSIDE the fit range. */
function ruleReproduces(season, seq, checkpoints) {
  const rounds = ruleBracket(season, seq);
  for (const r of checkpoints) {
    if (!sameSet(rounds[r].map(g => g.win), new Set(season.actual[r]))) return false;
  }
  return true;
}

/* How many of `rules` reproduce the checkpoints in at least one of `seasons`
 * (the seasons OUTSIDE the run), and the most seasons any one does --
 * the one statement about a search that its inputs cannot tune: "of the N
 * survivors, K reproduce any other season". Exact while rules x seasons is
 * within RULE_GEN_DIRECT applications (7,137 rules x 12 seasons is 0.3 s);
 * beyond that, on the simplest `checked` rules, and the caller says so. */
const RULE_GEN_DIRECT = 200000;
function ruleGeneralisation(rules, seasons, checkpoints) {
  const cps = new Set(checkpoints);
  const n = rules.length;
  if (!n || !seasons.length) return { n, checked: n, any: 0, best: 0 };
  const checked = Math.min(n, Math.max(1, Math.floor(RULE_GEN_DIRECT / seasons.length)));
  let any = 0, best = 0;
  for (let i = 0; i < checked; i++) {
    let k = 0;
    for (const sn of seasons) if (ruleReproduces(sn, rules[i], cps)) k++;
    if (k) any++;
    if (k > best) best = k;
  }
  return { n, checked, any, best };
}

/* The whole search job, pure, so it can run in a worker (app.js
 * runRuleSearchJob) or inline where there is none -- the same function
 * either way, never a second implementation. ruleRun over the played
 * seasons (ranked, by default, to prefer a rule built from specific
 * variables over an equally simple one leaning on a composite rating or a
 * rank -- see RULE_GENERAL_KEYS); the offered entries (distinct by the
 * bracket they give when a field exists, by sequence when it does not);
 * each entry's matches among the seasons OUTSIDE the run and its per-round
 * record over every played season; the generalisation of every survivor;
 * and the resolution of the rule a link names.
 *
 *   played       every complete played season, newest first
 *   here         the displayed season, or null while its field is pending
 *   keys         the criteria, sorted
 *   checkpoints  rounds whose winners a rule must reproduce (array)
 *   maxCriteria  distinct criteria a rule may use (never null on the page)
 *   want         a criterion sequence to find among the survivors, or null
 */
const RULE_OFFERED = 3;   // the simplest rule and two alternatives
function ruleSearchJob({ played, here, keys, checkpoints, maxCriteria, want }) {
  const cps = new Set(checkpoints);
  const res = ruleRun(played, keys, checkpoints, maxCriteria);
  // Outside the run -- skipped newer seasons included -- these are
  // descriptive: a match there never extends the run.
  const outside = played.filter(p => !res.run.includes(p.year));
  const apply = seq => {
    if (!here) return { rounds: null, picks: null };
    const rounds = ruleBracket(here, seq);
    return { rounds, picks: rounds.map(g => g.map(x => x.win)) };
  };
  const sigOf = (seq, b) => (here ? b.picks.map(g => g.join(',')).join('|') : seq.join(','));
  const entry = seq => {
    const hits = outside.filter(p => ruleReproduces(p, seq, cps)).map(p => p.year);
    const perRound = [...cps].sort((a, b) => a - b).map(r => ({ r, k: played.filter(p => ruleReproduces(p, seq, [r])).length, m: played.length }));
    const general = [...new Set(seq)].filter(k => RULE_GENERAL_KEYS.has(k)).length;
    return { seq, ...apply(seq), complexity: ruleComplexity(seq), general, matches: { k: hits.length, m: outside.length, years: hits }, perRound };
  };
  const offered = [], seen = new Set();
  for (const seq of res.rules) {
    const b = entry(seq);
    const sig = sigOf(seq, b);
    if (seen.has(sig)) continue;
    seen.add(sig);
    offered.push(b);
    if (offered.length >= RULE_OFFERED) break;
  }
  const gen = ruleGeneralisation(res.rules, outside, cps);
  // The rule a link names (or the one chosen before the season changed):
  // find it among the offered entries by sequence, then -- with a field --
  // by the bracket it gives, since the offered list is deduplicated by
  // picks; the entry then becomes the named rule itself, so the panel row
  // and the round labels carry the sequence the link names. Failing that,
  // if it survived at all, list it as one more. A named rule that did not
  // survive is reported (wantMissed), not quietly replaced.
  let chosen = 0, wantMissed = false;
  if (want) {
    const same = seq => seq.length === want.length && seq.every((k, i) => k === want[i]);
    let i = offered.findIndex(b => same(b.seq));
    if (i < 0 && res.rules.some(same)) {
      const b = entry(want);
      if (here) { const sig = sigOf(want, b); i = offered.findIndex(o => sigOf(o.seq, o) === sig); if (i >= 0) offered[i] = b; }
      if (i < 0) { offered.push(b); i = offered.length - 1; }
    }
    wantMissed = i < 0;
    chosen = Math.max(0, i);
  }
  return { run: res.run, skipped: res.skipped, stoppedAt: res.stoppedAt, nRules: res.rules.length, entries: offered, chosen, wantMissed, gen, nOutside: outside.length };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    fitLinear, fitQuality, crossValidate, scoreSpread, predictMargin,
    winProbFromMargin, knnPredict, normalCdf, studentTCdf, calibrate, clipProb, logLossFor,
    solve, stability, FIT, PROB_CLIP, causalWalkForward, CAL_PRIOR_STRENGTH,
    bracketAdvancementProbs, pairwiseCorrelations, trainingRows,
    reliabilityTable, RELIABILITY_EDGES, variableRecord, exclusionModels,
    rulePlay, ruleSequencesForSeason, ruleBracket, ruleReproduces, ruleComplexity, ruleComplexityOfCode, ruleGeneralCount, RULE_GENERAL_KEYS, decodeRule, ruleGeneralisation, RULE_GEN_DIRECT, ruleSearchJob, RULE_OFFERED, ruleRun, bitCount,
    RISK_ROUND_NAMES, DEFAULT_SCORING, POOL_FACTOR_THRESHOLD, makeRiskScorer,
    decideRiskWinner, riskGreedyBracket, riskExpectedPoints, solveByRisk,
  };
}
