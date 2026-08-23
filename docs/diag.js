/* Regression diagnostics for the spread model.
 *
 * WHAT THIS IS NOT
 * These numbers do not feed the bracket. The board is filled by the ridge fit in
 * fit.js; nothing here changes a single pick. This is a workbench for asking
 * whether the model that fills it is sound.
 *
 * WHY THE DIAGNOSTICS REFIT WITHOUT THE PENALTY
 * fit.js applies a ridge penalty. A penalised coefficient has no standard error
 * in the ordinary sense -- it is deliberately biased toward zero, so the usual
 * sampling-variance formula does not describe it and a p-value computed from it
 * would be a fiction. Everything below therefore refits by ORDINARY least
 * squares on the same rows and columns. That means:
 *
 *   - the coefficients here will not exactly match the equation on the board;
 *   - where they differ a lot, the ridge was doing real work, and that is
 *     itself the finding.
 *
 * THE CAVEAT NO TABLE CAN FIX
 * The user chooses the variables interactively, after seeing results. Every
 * p-value below is therefore conditional on a selection made with knowledge of
 * the data -- classic post-selection inference. A p-value earned that way is
 * optimistic and cannot be read as "this variable matters with probability
 * 1-p". It is a screening aid. The honest signal remains fold-to-fold
 * coefficient stability, which is computed in fit.js and shown on the board.
 */

/* ---------- linear algebra ---------- */

function matInverse(A) {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;   // singular: report, never fake
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (!f) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map(row => row.slice(n));
}

/* ---------- distributions ---------- */

/* Regularised incomplete beta, by the Lentz continued fraction.
 * Used for exact t-distribution tail probabilities. At the sample sizes here
 * (n ~ 800-950) the normal approximation would agree to about 1e-4, but exact
 * is cheap and removes a caveat. */
function betacf(a, b, x) {
  const FPMIN = 1e-30, EPS = 3e-12;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function lgamma(z) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += g[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (bt * betacf(a, b, x)) / a
    : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Two-sided p-value for a t statistic. */
function tPValue(t, df) {
  if (!isFinite(t) || df <= 0) return null;
  return betai(df / 2, 0.5, df / (df + t * t));
}

function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + z / 2);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** Upper-tail p for a chi-square with 1 degree of freedom. */
function chi2P1(x) {
  return x <= 0 ? 1 : erfc(Math.sqrt(x / 2));
}

/* ---------- the fit under examination ---------- */

function trainingRows(rows, asOf) {
  return asOf === null || asOf === undefined ? rows : rows.filter(r => r.y < asOf);
}

/* Ordinary least squares, no intercept, on the enabled columns.
 *
 * Returns everything the other diagnostics need, including (X'X)^-1, which is
 * what carries the standard errors and the variance inflation factors.
 */
function ols(rows, cols, asOf, withIntercept) {
  const used = trainingRows(rows, asOf);
  const k = cols.length + (withIntercept ? 1 : 0);
  if (!cols.length || used.length <= k + 1) return null;

  const design = r => {
    const row = cols.map(c => r.x[c]);
    if (withIntercept) row.push(1);
    return row;
  };

  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  let tss0 = 0, sumY = 0;
  for (const r of used) {
    const xr = design(r);
    for (let a = 0; a < k; a++) {
      Xty[a] += xr[a] * r.m;
      for (let b = a; b < k; b++) XtX[a][b] += xr[a] * xr[b];
    }
    tss0 += r.m * r.m;
    sumY += r.m;
  }
  for (let a = 0; a < k; a++) for (let b = a; b < k; b++) XtX[b][a] = XtX[a][b];

  const inv = matInverse(XtX);
  if (!inv) return { singular: true, n: used.length, k };

  const beta = inv.map((row, i) => row.reduce((s, v, j) => s + v * Xty[j], 0));

  const meanY = sumY / used.length;
  let rss = 0, tssMean = 0;
  const fitted = [], resid = [];
  for (const r of used) {
    const xr = design(r);
    let p = 0;
    for (let a = 0; a < k; a++) p += beta[a] * xr[a];
    const e = r.m - p;
    fitted.push(p); resid.push(e);
    rss += e * e;
    tssMean += (r.m - meanY) ** 2;
  }
  const df = used.length - k;
  const sigma2 = rss / df;

  return {
    beta, inv, XtX, rss, df, sigma2, n: used.length, k, cols, withIntercept,
    fitted, resid, meanY,
    // Both conventions, because they answer different questions and the gap
    // between them is the whole point of the intercept discussion.
    r2_about_zero: 1 - rss / tss0,
    r2_about_mean: 1 - rss / tssMean,
    tss0, tssMean,
  };
}

/* ---------- 1. significance ---------- */

function coefficientTable(fit, keys) {
  if (!fit || fit.singular) return null;
  return fit.beta.map((b, i) => {
    const se = Math.sqrt(fit.sigma2 * fit.inv[i][i]);
    const t = se > 0 ? b / se : null;
    const p = t === null ? null : tPValue(t, fit.df);
    const crit = 1.96;   // ~95% at these degrees of freedom
    return {
      key: fit.withIntercept && i === fit.beta.length - 1 ? '(intercept)' : keys[i],
      beta: b,
      se,
      t,
      p,
      ci: [b - crit * se, b + crit * se],
      significant: p !== null && p < 0.05,
    };
  });
}

/* ---------- 2. multicollinearity ---------- */

/* Variance inflation factor, in the form appropriate to a no-intercept model.
 *
 *   VIF_j = [(X'X)^-1]_jj * sum_i x_ij^2
 *
 * which is algebraically the usual 1/(1 - R2_j) with R2_j measured about ZERO
 * rather than about the column mean. That matters: the standard centred VIF
 * silently assumes an intercept is present to absorb the means. Using the
 * centred form on a through-the-origin design would report the wrong number.
 *
 * Reading: 1 means the column is orthogonal to the others. Above ~5 the
 * coefficient is being estimated from a small residual slice of the column and
 * is correspondingly unstable; above ~10 it is conventionally called severe.
 */
function vifTable(fit, keys) {
  if (!fit || fit.singular) return null;
  return fit.cols.map((_, i) => {
    const vif = fit.inv[i][i] * fit.XtX[i][i];
    return {
      key: keys[i],
      vif,
      // How much wider the confidence interval is than it would be if this
      // column were uncorrelated with the rest -- the practical cost.
      seInflation: Math.sqrt(vif),
      severity: vif >= 10 ? 'severe' : vif >= 5 ? 'high' : vif >= 2.5 ? 'moderate' : 'low',
    };
  });
}

/* ---------- 3. sign and magnitude sanity ---------- */

/* Every z-score is sign-corrected upstream so that HIGHER IS BETTER. A negative
 * coefficient therefore contradicts the construction of its own column, and is
 * worth an explanation rather than a shrug. The usual explanation here is
 * collinearity -- the variable is cancelling part of a near-duplicate -- which
 * is why the univariate coefficient is shown alongside. A variable that is
 * positive alone and negative in company has not reversed its effect; it has
 * been assigned someone else's.
 */
function signTable(rows, cols, asOf, fit, keys) {
  if (!fit || fit.singular) return null;
  return fit.cols.map((c, i) => {
    const solo = ols(rows, [c], asOf, false);
    const alone = solo && !solo.singular ? solo.beta[0] : null;
    const joint = fit.beta[i];
    const flipped = alone !== null && Math.sign(alone) !== Math.sign(joint) && Math.abs(joint) > 0.05;
    return {
      key: keys[i],
      alone,
      joint,
      expected: '+',                       // by construction of the z-scores
      wrongSign: joint < -0.05,
      flipped,
      // A coefficient far larger than the spread of the target is not an effect
      // anyone can act on; it is one side of a cancelling pair.
      implausible: Math.abs(joint) > 15,
    };
  });
}

/* ---------- 4. residuals ---------- */

function residualDiagnostics(fit) {
  if (!fit || fit.singular) return null;
  const { fitted, resid, n } = fit;
  const s = Math.sqrt(fit.rss / n);

  // Breusch-Pagan, run against the ABSOLUTE fitted value.
  //
  // The magnitude matters and is not a stylistic choice. This model is
  // antisymmetric: swapping the two teams negates the prediction while leaving
  // the error spread untouched, so variance is an EVEN function of the fitted
  // value -- Var(e | yhat) = Var(e | -yhat) by construction. Regressing squared
  // residuals on the SIGNED fitted value therefore estimates a slope that must
  // be about zero however severe the heteroscedasticity is, and the test has
  // essentially no power against the only pattern this design can produce.
  // Verified: on data built with error scaled by |x|, the signed version
  // returned p = 0.92 while the magnitude version returns p < 1e-15.
  //
  // A significant result means the error spread grows (or shrinks) with the
  // size of the predicted margin -- the model is systematically more wrong
  // about mismatches than about close games, or vice versa, so the single sigma
  // used to turn a margin into a probability is wrong at one end of the range.
  const absFitted = fitted.map(Math.abs);
  const mf = absFitted.reduce((a, b) => a + b, 0) / n;
  const e2 = resid.map(e => e * e);
  const me2 = e2.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (absFitted[i] - mf) * (e2[i] - me2);
    sxx += (absFitted[i] - mf) ** 2;
    syy += (e2[i] - me2) ** 2;
  }
  const r2aux = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : 0;
  const bpStat = n * r2aux;
  const bpP = chi2P1(bpStat);
  // Sign of the relationship, so the UI can say which end is underserved.
  const bpSlope = sxx > 0 ? sxy / sxx : 0;

  // Non-linearity: split the fitted range into bands and look at the mean
  // residual in each. A model that is linear in these inputs should scatter
  // around zero everywhere; a curve shows up as a run of same-signed means.
  const BANDS = 8;
  const lo = Math.min(...fitted), hi = Math.max(...fitted);
  const width = (hi - lo) / BANDS || 1;
  const bands = Array.from({ length: BANDS }, () => ({ n: 0, sum: 0, sumAbs: 0, lo: 0, hi: 0 }));
  for (let i = 0; i < BANDS; i++) { bands[i].lo = lo + i * width; bands[i].hi = lo + (i + 1) * width; }
  for (let i = 0; i < n; i++) {
    let b = Math.min(BANDS - 1, Math.floor((fitted[i] - lo) / width));
    if (b < 0) b = 0;
    bands[b].n++; bands[b].sum += resid[i]; bands[b].sumAbs += Math.abs(resid[i]);
  }
  const bandStats = bands.map(b => ({
    lo: b.lo, hi: b.hi, n: b.n,
    meanResid: b.n ? b.sum / b.n : null,
    meanAbsResid: b.n ? b.sumAbs / b.n : null,
  }));

  const std = resid.map(e => e / s);
  const outliers = [];
  for (let i = 0; i < n; i++) {
    if (Math.abs(std[i]) > 3) outliers.push({ i, fitted: fitted[i], resid: resid[i], z: std[i] });
  }
  outliers.sort((a, b) => Math.abs(b.z) - Math.abs(a.z));

  return {
    sigma: s,
    points: fitted.map((f, i) => [f, resid[i]]),
    bands: bandStats,
    heteroscedasticity: {
      statistic: bpStat,
      p: bpP,
      significant: bpP < 0.05,
      // Positive: bigger predicted margins carry bigger errors, so the board is
      // over-confident about mismatches. Negative: the reverse.
      widensWithMargin: bpSlope > 0,
    },
    outliers: outliers.slice(0, 10),
    outlierCount: outliers.length,
    outlierRate: outliers.length / n,
    // Under normal errors ~0.27% of points sit beyond 3 sigma. Materially more
    // means heavy tails: blowouts the model cannot see coming.
    expectedOutlierRate: 0.0027,
    meanResid: resid.reduce((a, b) => a + b, 0) / n,
  };
}

/* ---------- 5. the no-intercept assumption ---------- */

/* Fit the same columns WITH a constant and test whether it is distinguishable
 * from zero.
 *
 * The design argument for excluding it is strong and does not depend on this
 * test: the rows are differentials, so swapping the two teams must negate the
 * prediction exactly. A constant breaks that -- it would say one side wins by c
 * points before anyone looks at the teams, which is a statement about row order
 * rather than basketball.
 *
 * The test still earns its place, because it says what that assumption COSTS.
 * A significant intercept would mean the rows carry a systematic asymmetry the
 * model is refusing to fit -- most likely an artefact of the seed-first
 * ordering rather than a reason to add a constant.
 *
 * The R2 comparison is the other half. Through the origin, "R2" is routinely
 * quoted about zero, which uses a different and much larger baseline than the
 * about-the-mean figure everyone assumes. Both are shown so the number cannot
 * be read as the familiar one by accident.
 */
function interceptTest(rows, cols, asOf, keys) {
  const without = ols(rows, cols, asOf, false);
  const with_ = ols(rows, cols, asOf, true);
  if (!without || !with_ || without.singular || with_.singular) return null;

  const i = with_.beta.length - 1;
  const se = Math.sqrt(with_.sigma2 * with_.inv[i][i]);
  const t = se > 0 ? with_.beta[i] / se : null;
  const p = t === null ? null : tPValue(t, with_.df);

  // F test of the single restriction (intercept = 0).
  const f = ((without.rss - with_.rss) / 1) / with_.sigma2;

  return {
    intercept: with_.beta[i],
    se, t, p,
    significant: p !== null && p < 0.05,
    fStat: f,
    rssWithout: without.rss,
    rssWith: with_.rss,
    rmseWithout: Math.sqrt(without.rss / without.n),
    rmseWith: Math.sqrt(with_.rss / with_.n),
    r2ZeroWithout: without.r2_about_zero,
    r2MeanWithout: without.r2_about_mean,
    r2MeanWith: with_.r2_about_mean,
    meanY: without.meanY,
    coefShift: without.beta.map((b, j) => ({ key: keys[j], without: b, with: with_.beta[j] })),
  };
}

/* ---------- 6. baselines ---------- */

/* Nothing above means anything without something to beat.
 *
 * PREDICT ZERO is the honest null for this design. The model claims two teams
 * differ by some number of points; the null claims they are even. It is also
 * exactly the baseline that "R2 about zero" scores against, which is why that
 * figure is the defensible one here.
 *
 * PREDICT A CONSTANT is the baseline "R2 about the mean" scores against, and it
 * is a cheat in this setting: the rows are ordered better-seed-first, so the
 * mean margin is positive, and a model that always predicts +6.8 is exploiting
 * the row layout rather than the teams. The constant is taken from TRAINING
 * seasons only, so it is at least not also peeking at the answer.
 *
 * PICK THE BETTER SEED needs no fit at all, and because rows are seed-ordered
 * it is simply "always pick the first team".
 */
function baselines(rows, cols, asOf, fit) {
  const used = trainingRows(rows, asOf);
  if (!used.length) return null;

  const score = predict => {
    let sse = 0, sae = 0, sst = 0, correct = 0, n = 0;
    for (const r of used) {
      const p = predict(r);
      sse += (r.m - p) ** 2;
      sae += Math.abs(r.m - p);
      sst += r.m * r.m;
      if (r.m !== 0 && ((p > 0) === (r.m > 0))) correct++;
      n++;
    }
    return { rmse: Math.sqrt(sse / n), mae: sae / n, r2: 1 - sse / sst, accuracy: correct / n };
  };

  const meanM = used.reduce((a, r) => a + r.m, 0) / used.length;

  const out = [
    { name: 'Predict a dead heat (0 points)', note: 'the null this model\'s R² is measured against', ...score(() => 0) },
    { name: `Always favour the better seed by ${meanM.toFixed(1)}`, note: 'the constant-only model; exploits row order', ...score(() => meanM) },
  ];
  if (fit && !fit.singular) {
    out.push({
      name: `Fitted model (${fit.cols.length} variable${fit.cols.length > 1 ? 's' : ''}, in-sample)`,
      note: 'ordinary least squares on the training seasons',
      ...score(r => {
        let p = 0;
        for (let a = 0; a < fit.cols.length; a++) p += fit.beta[a] * r.x[fit.cols[a]];
        return p;
      }),
    });
  }
  return out;
}

/* ---------- assembly ---------- */

function diagnose(rows, cols, keys, asOf) {
  const fit = ols(rows, cols, asOf, false);
  if (!fit) return { error: 'Not enough data for these variables.' };
  if (fit.singular) {
    return {
      error:
        'The selected variables are perfectly collinear — one is an exact linear ' +
        'combination of the others, so the coefficients are not identified. ' +
        'Switch one off.',
    };
  }
  return {
    fit,
    coefficients: coefficientTable(fit, keys),
    vif: vifTable(fit, keys),
    signs: signTable(rows, cols, asOf, fit, keys),
    residuals: residualDiagnostics(fit),
    intercept: interceptTest(rows, cols, asOf, keys),
    baselines: baselines(rows, cols, asOf, fit),
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    diagnose, ols, coefficientTable, vifTable, signTable,
    residualDiagnostics, interceptTest, baselines,
    tPValue, chi2P1, betai, matInverse,
  };
}
