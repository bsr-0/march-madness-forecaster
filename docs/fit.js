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

/* Blend the model's probability with a historical seed-matchup base rate.
 *
 * WHY A BLEND AND NOT A FEATURE. As a feature the fit decides how much the
 * base rate matters and the user cannot move it; as a blend the user sets the
 * weight and can see the board respond. That is the point of the control --
 * "how much do I trust history over the variables for THIS pairing" is a
 * question the fit cannot answer for someone else.
 *
 * ANTISYMMETRY SURVIVES THIS, which is not automatic and is why the blend is
 * a plain convex combination. Swapping the teams sends p -> 1-p and q -> 1-q,
 * so (1-w)p + wq -> (1-w)(1-p) + w(1-q) = 1 - [(1-w)p + wq]. Any blend that
 * is not affine in both arguments would break the guarantee the whole board
 * rests on -- see the antisymmetry checks in tests/test_calibration.js.
 *
 * weight 0 returns the model untouched, so the control is a no-op at rest.
 */
function blendWithPrior(pModel, pPrior, weight) {
  if (!isFinite(pPrior) || !(weight > 0)) return pModel;
  const w = Math.min(1, Math.max(0, weight));
  return clipProb((1 - w) * pModel + w * pPrior);
}

function winProbFromMargin(margin, sigma, cal) {
  const a = cal && isFinite(cal.a) ? cal.a : 1;
  const nu = cal ? cal.nu : Infinity;
  return clipProb(studentTCdf(a * margin / Math.max(sigma, 1e-6), nu));
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    fitLinear, fitQuality, crossValidate, scoreSpread, predictMargin,
    winProbFromMargin, blendWithPrior, knnPredict, normalCdf, studentTCdf, calibrate, clipProb, logLossFor,
    solve, stability, FIT, PROB_CLIP,
  };
}
