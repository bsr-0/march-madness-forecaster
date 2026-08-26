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
const CANONICAL_KEYS = [
  'barthag', 't_rank', 'massey_avg_rank', 'sos_avg_opp_barthag',
  'adj_offensive_efficiency', 'adj_defensive_efficiency', 'adj_tempo',
  'effective_fg_pct', 'three_pt_pct', 'three_pt_rate',
  'offensive_reb_rate', 'turnover_rate',
];

const WIDTH_SWEEP = [3, 8, 12, 20, 30];
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

function score(pred, cal) {
  let ll = 0, brier = 0, n = 0, correct = 0, sse = 0, sae = 0, pinned = 0;
  let closest = 1;
  for (const r of pred) {
    const raw = F.studentTCdf(cal.a * r.margin / r.sigma, cal.nu);
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
    console.log(`  calibration   a=${b.calibration.a} nu=${b.calibration.nu}`);
    console.log(`  log loss      ${b.calibrated.logLoss.toFixed(4)}  (uncalibrated ${b.uncalibrated.logLoss.toFixed(4)})`);
    console.log(`  Brier         ${b.calibrated.brier.toFixed(4)}  (uncalibrated ${b.uncalibrated.brier.toFixed(4)})`);
    console.log(`  accuracy      ${(b.calibrated.accuracy * 100).toFixed(2)}%`);
    return;
  }

  if (has('--compare')) {
    const to = val('--to', DEFAULT_OUT);
    if (!fs.existsSync(to)) { console.error(`no baseline at ${to} -- run --freeze first`); process.exit(1); }
    const was = JSON.parse(fs.readFileSync(to, 'utf8'));
    const now_keys = CANONICAL_KEYS.filter(k => !dropKeys.includes(k));
    if ((was.canonicalKeys || []).length !== now_keys.length) {
      console.log(`  KEY SET DIFFERS: baseline ${was.canonicalKeys.length} keys -> now ${now_keys.length}`);
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
    for (const [label, key, lower] of M) {
      console.log(`  ${label.padEnd(10)} ${was.calibrated[key].toFixed(4)} -> ${now.calibrated[key].toFixed(4)}   ${fmtDelta(now.calibrated[key], was.calibrated[key], lower)}`);
    }
    console.log(`\n  aggregate calibration gap/SE ${was.reliability.aggregate.gapSE} -> ${now.reliability.aggregate.gapSE}`);
    console.log(`  pinned predictions           ${was.calibrated.pinnedPastTolerance} -> ${now.calibrated.pinnedPastTolerance}`);
    return;
  }

  console.error('usage: node scripts/model_baseline.js --freeze | --compare');
  process.exit(1);
}

main();
