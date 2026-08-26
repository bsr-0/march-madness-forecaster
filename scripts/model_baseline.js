/* Freeze and compare the browser model's walk-forward performance.
 *
 * WHY THIS EXISTS. Every change to the model -- a new feature, a wider training
 * population, a different link -- is claimed to be an improvement. This file is
 * how that claim gets checked instead of asserted. Freeze once, then re-run
 * after each change and diff. Without a frozen artifact there is no way to tell
 * a fix from a regression, because the numbers move on every rebuild of the
 * underlying data and memory of "it was about 0.45" is not evidence.
 *
 * VARIABLES ARE SELECTED BY KEY, NEVER BY INDEX. training.json's `keys` array
 * is positional, and variables get inserted into the middle of VARIABLES as the
 * feature set grows -- three were added in the middle of the Overall and Roster
 * groups recently. An index-based baseline would silently start measuring a
 * different model the next time that happens, and the comparison would look
 * like a regression caused by whatever else changed that day. Keys are stable;
 * indices are not.
 *
 * Usage:
 *   node scripts/model_baseline.js --freeze [--out FILE]
 *   node scripts/model_baseline.js --compare [--to FILE]
 */

const fs = require('fs');
const path = require('path');

const REPO = path.join(__dirname, '..');
const F = require(path.join(REPO, 'docs', 'fit.js'));
const TRAINING = path.join(REPO, 'docs', 'data', 'training.json');
const DEFAULT_OUT = path.join(REPO, 'artifacts', 'model_baseline.json');

/* The canonical evaluation set: the twelve variables the calibration work was
 * measured on. Near the accuracy peak (~78% walk-forward) and comfortably
 * clear of the 30-variable regime where half the coefficients flip sign
 * between folds. Held fixed so successive runs are comparable to each other,
 * which is the entire point -- this is a measuring stick, not a
 * recommendation about which variables a user should enable. */
// massey_avg_rank was REMOVED here on 2026-08-25, deliberately and not as a
// cleanup. audit_opponent_adjustment measured it at partial r = -0.744 against
// conference strength controlling for team quality -- by a wide margin the
// worst confound of the twelve, and worse than any raw per-game rate. It is
// not opponent-adjustable: it pools other people's ranking systems, several of
// them RPI-like, so the conference reward is baked into inputs this repo does
// not control. It is also ordinal, so it is on the wrong scale for a margin
// model regardless.
//
// Dropping it measured FREE against the 12-key baseline (archived at
// artifacts/baselines/baseline_12key_pre_massey_drop.json):
//   log loss 0.4514 -> 0.4513   Brier 0.1470 -> 0.1469   RMSE 10.3256 -> 10.3188
//   accuracy 0.7804 -> 0.7778, which is 2 games of 756 and inside noise.
// Evidenced rather than inferred, which is why it is gone rather than flagged.
const CANONICAL_KEYS = [
  'barthag', 't_rank', 'sos_avg_opp_barthag',
  'adj_offensive_efficiency', 'adj_defensive_efficiency', 'adj_tempo',
  'effective_fg_pct', 'three_pt_pct', 'three_pt_rate',
  'offensive_reb_rate', 'turnover_rate',
];

const WIDTH_SWEEP = [3, 8, 12, 20, 30];

// Shrinkage weight for walk-forward calibration, in units of observations.
// One season is 63 games, so a year with only one prior season of residuals
// sits halfway between its own fit and the uninformative a = 1.
//
// WHY SHRINK RATHER THAN EXCLUDE THE COLD-START YEARS. 2014 and 2015 have too
// few prior test years to calibrate from. Dropping them would change what
// "756 predictions" means and put a permanent asterisk on every later
// comparison. Shrinking toward a = 1 keeps the row set fixed and makes the
// cold-start cost visible in the number instead of hiding it by deletion:
// 2014 falls back to exactly uncalibrated, and the prior fades as residuals
// accumulate.
//
// a IS SHRUNK TOWARD 1, NOT TOWARD THE GLOBAL FIT, because the global fit is
// computed from all 756 including the year being scored -- the very leak this
// replaces.
const CAL_PRIOR_STRENGTH = 63;
const BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001];

function resolveCols(keys, wanted) {
  const cols = [], missing = [];
  for (const k of wanted) {
    const i = keys.indexOf(k);
    if (i < 0) missing.push(k); else cols.push(i);
  }
  if (missing.length) {
    throw new Error(
      `training.json is missing baseline variables: ${missing.join(', ')}.\n` +
      `The baseline is defined by key, so a renamed or dropped variable must be ` +
      `dealt with deliberately rather than silently changing what is measured.`
    );
  }
  return cols;
}

/* Walk-forward predictions: for each test season, fit on strictly earlier
 * seasons and predict that season's games. sigma travels with each row because
 * it is refit per fold; collapsing it to a global value would be a different
 * (and wrong) model. */
function walkForward(rows, cols, years, minYear) {
  const out = [];
  for (const y of years.filter(v => v >= minYear)) {
    const f = F.fitLinear(rows, cols, y);
    if (!f.ok) continue;
    for (const r of rows.filter(v => v.y === y)) {
      out.push({ y, m: r.m, margin: F.predictMargin(f.beta, cols, r.x), sigma: f.sigma });
    }
  }
  return out;
}

/* Per-year calibration fit ONLY on strictly earlier test years.
 *
 * The global calibrate() is fit on all 756 walk-forward predictions and then
 * used to score those same 756. The margins are out of sample; the calibration
 * is not, and the resulting log loss is flattered by 0.00181 (measured on the
 * 630 warm-year predictions, which excludes cold-start from the comparison).
 * Small, but a self-graded constant is not something to leave inside a frozen
 * baseline that later work is measured against.
 *
 * Returns a lookup year -> {a, nu}. Per-year a is NOT used raw: bootstrapped
 * per-year CIs are enormous (2014 [0.54, 2.33]) and 11 of 12 contain the
 * global value, with no significant trend (-0.0034 +/- 0.0927 per year), so
 * the year-to-year spread is sampling noise on ~63 binary outcomes rather than
 * signal. Shrinkage is what keeps that noise out of the scored numbers.
 */
function walkForwardCalibration(pred, years) {
  const byYear = {};
  for (const y of years) {
    const prior = pred.filter(r => r.y < y);
    if (!prior.length) {
      byYear[y] = { a: 1, nu: Infinity, priorN: 0 };
      continue;
    }
    const fit = F.calibrate(prior.map(r => ({ m: r.m, p: r.margin, sigma: r.sigma })));
    const n = prior.length;
    const w = n / (n + CAL_PRIOR_STRENGTH);
    byYear[y] = { a: w * fit.a + (1 - w) * 1, nu: fit.nu, priorN: n };
  }
  return byYear;
}

function score(pred, cal) {
  let ll = 0, brier = 0, n = 0, correct = 0, sse = 0, sae = 0, pinned = 0;
  let closest = 1;
  for (const r of pred) {
    // cal may be one object or a per-year lookup, so a walk-forward
    // calibration can be scored by the same function as a global one.
    const c = typeof cal === 'function' ? cal(r) : cal;
    const raw = F.studentTCdf(c.a * r.margin / r.sigma, c.nu);
    const p = F.clipProb(raw);
    sse += (r.m - r.margin) ** 2;
    sae += Math.abs(r.m - r.margin);
    if (r.m === 0) continue;
    const y = r.m > 0 ? 1 : 0;
    ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    brier += (p - y) ** 2;
    if ((r.margin > 0) === (r.m > 0)) correct++;
    if (p >= 1 - 1e-4 || p <= 1e-4) pinned++;
    closest = Math.min(closest, Math.min(raw, 1 - raw));
    n++;
  }
  return {
    n,
    logLoss: ll / n,
    brier: brier / n,
    accuracy: correct / n,
    rmse: Math.sqrt(sse / pred.length),
    mae: sae / pred.length,
    pinnedPastTolerance: pinned,
    closestApproachToBoundPreClip: closest,
  };
}

/* Reliability table. Reported with the standard error of each bin, because a
 * gap is meaningless without one -- a 6-point miss on 84 coin-flip games is
 * noise and a 6-point miss on 400 is not, and the table should make that
 * impossible to confuse. */
function reliability(pred, cal) {
  const P = [];
  for (const r of pred) {
    if (r.m === 0) continue;
    P.push({ p: F.clipProb(F.studentTCdf(cal.a * r.margin / r.sigma, cal.nu)), won: r.m > 0 });
  }
  const table = [];
  for (let i = 0; i < BINS.length - 1; i++) {
    const b = P.filter(o => o.p >= BINS[i] && o.p < BINS[i + 1]);
    if (!b.length) { table.push({ bin: [BINS[i], Math.min(BINS[i + 1], 1)], n: 0 }); continue; }
    const pm = b.reduce((a, o) => a + o.p, 0) / b.length;
    const am = b.filter(o => o.won).length / b.length;
    const se = Math.sqrt(pm * (1 - pm) / b.length);
    table.push({
      bin: [BINS[i], Math.min(BINS[i + 1], 1)], n: b.length,
      predicted: +pm.toFixed(4), actual: +am.toFixed(4),
      gap: +(am - pm).toFixed(4), gapSE: se > 0 ? +((am - pm) / se).toFixed(2) : null,
    });
  }
  // Aggregate count check: immune to where the bin edges fall, which is the
  // failure mode that made the 0.5-0.6 bin look like a finding.
  const expected = P.reduce((a, o) => a + o.p, 0);
  const observed = P.filter(o => o.won).length;
  const sd = Math.sqrt(P.reduce((a, o) => a + o.p * (1 - o.p), 0));
  return { table, aggregate: { n: P.length, expected: +expected.toFixed(1), observed, gapSE: +((observed - expected) / sd).toFixed(2) } };
}

// Variant support for the confound work: measure a key set against the frozen
// baseline WITHOUT editing CANONICAL_KEYS. Editing the constant would silently
// redefine what "the baseline" means, so the next --compare would diff two
// different models and report the difference as a change in performance. The
// drop list is echoed into the artefact so a variant can never be mistaken for
// the baseline after the fact.
function build(dropKeys = []) {
  const t = JSON.parse(fs.readFileSync(TRAINING, 'utf8'));
  const wanted = CANONICAL_KEYS.filter(k => !dropKeys.includes(k));
  if (dropKeys.length) {
    const unknown = dropKeys.filter(k => !CANONICAL_KEYS.includes(k));
    if (unknown.length) throw new Error(`--drop names non-baseline keys: ${unknown.join(', ')}`);
    console.log(`VARIANT: dropped ${dropKeys.join(', ')} -> ${wanted.length} of ${CANONICAL_KEYS.length} keys`);
  }
  const cols = resolveCols(t.keys, wanted);
  const pred = walkForward(t.games, cols, t.years, F.FIT.MIN_TEST_YEAR);
  const cal = F.calibrate(pred.map(r => ({ m: r.m, p: r.margin, sigma: r.sigma })));
  const uncal = { a: 1, nu: Infinity };
  const wfCal = walkForwardCalibration(pred, t.years.filter(y => y >= F.FIT.MIN_TEST_YEAR));
  const wfLookup = r => wfCal[r.y];
  // warm years are those with enough prior residuals for the prior to have
  // faded; reported separately so the cold-start cost is visible, not hidden.
  const warmYears = Object.keys(wfCal).filter(y => wfCal[y].priorN >= 100).map(Number);
  const warm = pred.filter(r => warmYears.includes(r.y));

  const sweep = {};
  for (const n of WIDTH_SWEEP) {
    if (n > t.keys.length) continue;
    const c = Array.from({ length: n }, (_, i) => i);
    const pw = walkForward(t.games, c, t.years, F.FIT.MIN_TEST_YEAR);
    const cw = F.calibrate(pw.map(r => ({ m: r.m, p: r.margin, sigma: r.sigma })));
    const s = score(pw, cw);
    sweep['first' + n] = { logLoss: +s.logLoss.toFixed(4), brier: +s.brier.toFixed(4), accuracy: +s.accuracy.toFixed(4) };
  }

  return {
    schema: 2,
    frozenAt: new Date().toISOString(),
    note: 'Walk-forward baseline for the browser model. Compare with --compare.',
    trainingSource: { games: t.games.length, seasons: t.years.length, years: t.years, nKeys: t.keys.length },
    canonicalKeys: wanted,
    droppedKeys: dropKeys,
    minTestYear: F.FIT.MIN_TEST_YEAR,
    calibration: { a: +cal.a.toFixed(4), nu: cal.nu === Infinity ? 'Infinity' : cal.nu },
    // The headline for any calibration claim. Fit strictly on earlier years,
    // shrunk toward a = 1 while residuals are scarce.
    calibratedWalkForward: score(pred, wfLookup),
    calibratedWalkForwardWarm: score(warm, wfLookup),
    walkForwardCalibration: Object.fromEntries(
      Object.entries(wfCal).map(([y, c]) => [y, { a: +c.a.toFixed(4), nu: c.nu === Infinity ? 'Infinity' : c.nu, priorN: c.priorN }])
    ),
    calibrated: score(pred, cal),
    uncalibrated: score(pred, uncal),
    reliability: reliability(pred, cal),
    widthSweep: sweep,
    // The predictions themselves, so a later change can be diffed game by game
    // rather than only in aggregate -- an aggregate can hold still while
    // individual predictions move a great deal.
    predictions: pred.map(r => ({
      y: r.y, m: r.m, margin: +r.margin.toFixed(4), sigma: +r.sigma.toFixed(4),
    })),
  };
}

function fmtDelta(now, was, lowerIsBetter) {
  if (was === undefined || was === null) return '   (new)';
  const d = now - was;
  if (Math.abs(d) < 1e-9) return '    same';
  const better = lowerIsBetter ? d < 0 : d > 0;
  return `${d >= 0 ? '+' : ''}${d.toFixed(4)} ${better ? 'better' : 'WORSE'}`;
}

function main() {
  const argv = process.argv.slice(2);
  const has = f => argv.includes(f);
  const val = (f, dflt) => { const i = argv.indexOf(f); return i >= 0 ? argv[i + 1] : dflt; };

  const dropArg = val('--drop', '');
  const dropKeys = dropArg ? dropArg.split(',').map(s => s.trim()).filter(Boolean) : [];

  if (has('--freeze')) {
    const out = val('--out', DEFAULT_OUT);
    const b = build(dropKeys);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, JSON.stringify(b, null, 2));
    console.log(`froze baseline -> ${path.relative(REPO, out)}`);
    console.log(`  ${b.calibrated.n} walk-forward predictions over ${b.trainingSource.seasons} seasons`);
    console.log(`  calibration   a=${b.calibration.a} nu=${b.calibration.nu} (global, self-graded)`);
    // The walk-forward figure is the headline for any calibration claim: the
    // global one is fit on the same predictions it scores. The warm subset
    // drops the cold-start years from the COMPARISON only; they stay in the
    // row set so "756" never changes meaning.
    console.log(`  log loss      ${b.calibratedWalkForwardWarm.logLoss.toFixed(5)}  walk-forward, warm ${b.calibratedWalkForwardWarm.n}`);
    console.log(`                ${b.calibratedWalkForward.logLoss.toFixed(5)}  walk-forward, all ${b.calibratedWalkForward.n} (incl. cold start)`);
    console.log(`                ${b.calibrated.logLoss.toFixed(5)}  global a  (uncalibrated ${b.uncalibrated.logLoss.toFixed(4)})`);
    console.log(`  Brier         ${b.calibratedWalkForwardWarm.brier.toFixed(5)}  walk-forward warm  (global ${b.calibrated.brier.toFixed(4)})`);
    console.log(`  accuracy      ${(b.calibrated.accuracy * 100).toFixed(2)}%`);
    return;
  }

  if (has('--compare')) {
    const to = val('--to', DEFAULT_OUT);
    if (!fs.existsSync(to)) { console.error(`no baseline at ${to} -- run --freeze first`); process.exit(1); }
    const was = JSON.parse(fs.readFileSync(to, 'utf8'));

    // A KEY-SET CHANGE IS NOT A PERFORMANCE CHANGE, AND MUST NOT READ AS ONE.
    // The whole point of selecting by key rather than index is that a dropped
    // or renamed variable cannot silently reindex into a different model. That
    // guarantee is worth nothing if --compare then diffs a 12-key baseline
    // against an 11-key variant and prints the difference as "better" or
    // "WORSE" with no indication the models differ. So this refuses outright.
    // --allow-key-change is the deliberate acknowledgement, and it is exactly
    // how a measured drop should be recorded: on purpose, in the command.
    const nowKeys = CANONICAL_KEYS.filter(k => !dropKeys.includes(k));
    const wasKeys = was.canonicalKeys || [];
    const added = nowKeys.filter(k => !wasKeys.includes(k));
    const removed = wasKeys.filter(k => !nowKeys.includes(k));
    if ((added.length || removed.length) && !has('--allow-key-change')) {
      console.error(
        `\nREFUSING TO COMPARE: the key set differs from the baseline.\n` +
        (removed.length ? `  removed: ${removed.join(', ')}\n` : '') +
        (added.length ? `  added:   ${added.join(', ')}\n` : '') +
        `  baseline ${wasKeys.length} keys -> now ${nowKeys.length}\n\n` +
        `These are different models, so a metric difference is not a change in\n` +
        `performance. Re-run with --allow-key-change to compare deliberately.\n`
      );
      process.exit(2);
    }
    if (added.length || removed.length) {
      console.log(`  KEY SET DIFFERS (acknowledged): ${wasKeys.length} -> ${nowKeys.length} keys`);
      if (removed.length) console.log(`    removed: ${removed.join(', ')}`);
      if (added.length) console.log(`    added:   ${added.join(', ')}`);
    }
    const now = build(dropKeys);
    console.log(`comparing against ${path.relative(REPO, to)} (frozen ${was.frozenAt})\n`);
    console.log(`  rows        ${was.trainingSource.games} -> ${now.trainingSource.games}`);
    console.log(`  predictions ${was.calibrated.n} -> ${now.calibrated.n}`);
    console.log(`  calibration a=${was.calibration.a} nu=${was.calibration.nu}  ->  a=${now.calibration.a} nu=${now.calibration.nu}\n`);
    const M = [
      ['log loss', 'logLoss', true], ['Brier', 'brier', true],
      ['accuracy', 'accuracy', false], ['RMSE', 'rmse', true], ['MAE', 'mae', true],
    ];
    // WALK-FORWARD FIRST, because it is the honest number. The global block
    // below is retained for continuity with older baselines, but a calibration
    // constant fit on the same predictions it scores flatters log loss by
    // ~0.0018 and should not be what a change is judged against.
    if (was.calibratedWalkForwardWarm && now.calibratedWalkForwardWarm) {
      console.log('  walk-forward calibration (headline, warm years):');
      for (const [label, key, lower] of M) {
        const a = was.calibratedWalkForwardWarm[key], b = now.calibratedWalkForwardWarm[key];
        console.log(`    ${label.padEnd(10)} ${a.toFixed(5)} -> ${b.toFixed(5)}   ${fmtDelta(b, a, lower)}`);
      }
      console.log('  global calibration (self-graded, for continuity):');
    } else {
      console.log('  baseline predates walk-forward calibration; global only:');
    }
    for (const [label, key, lower] of M) {
      console.log(`    ${label.padEnd(10)} ${was.calibrated[key].toFixed(4)} -> ${now.calibrated[key].toFixed(4)}   ${fmtDelta(now.calibrated[key], was.calibrated[key], lower)}`);
    }
    console.log(`\n  aggregate calibration gap/SE ${was.reliability.aggregate.gapSE} -> ${now.reliability.aggregate.gapSE}`);
    console.log(`  pinned predictions           ${was.calibrated.pinnedPastTolerance} -> ${now.calibrated.pinnedPastTolerance}`);
    return;
  }

  console.error('usage: node scripts/model_baseline.js --freeze | --compare');
  process.exit(1);
}

main();
