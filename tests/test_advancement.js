/* Checks for bracketAdvancementProbs() in docs/fit.js.
 *
 * This is the recursive subtree-merge that turns pairwise game probabilities
 * into each team's chance of reaching every round of a REAL bracket -- not a
 * seed-based table (src/prediction/seed_probabilities.py answers a different
 * question: base rates by seed number, ignoring who is actually on each side
 * of the draw). Wrong here means every "chance of reaching the Sweet 16"
 * number on the page is wrong, silently, because nothing else in the pipeline
 * checks it against a real tournament outcome.
 *
 * Run: node tests/test_advancement.js
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

console.log('bracket advancement');

check('a 2-team bracket reduces to the pairwise probability itself', () => {
  const p = F.bracketAdvancementProbs([0, 1], (a, b) => (a === 0 && b === 1 ? 0.7 : 0.3));
  close(p[0][0], 0.7, 1e-9);
  close(p[1][0], 0.3, 1e-9);
});

check('a 4-team bracket matches a hand-worked calculation', () => {
  // A-B and C-D in round 1, winners meet in round 2.
  const W = {
    '0,1': 0.6, '1,0': 0.4,
    '2,3': 0.3, '3,2': 0.7,
    '0,2': 0.55, '2,0': 0.45,
    '0,3': 0.65, '3,0': 0.35,
    '1,2': 0.5, '2,1': 0.5,
    '1,3': 0.6, '3,1': 0.4,
  };
  const winProb = (a, b) => W[`${a},${b}`];
  const p = F.bracketAdvancementProbs([0, 1, 2, 3], winProb);

  // Round 1 is just the direct pairwise probability.
  close(p[0][0], 0.6, 1e-9, 'A round 1');
  close(p[1][0], 0.4, 1e-9, 'B round 1');
  close(p[2][0], 0.3, 1e-9, 'C round 1');
  close(p[3][0], 0.7, 1e-9, 'D round 1');

  // Round 2 (champion), worked by hand: A only gets there by beating B, then
  // faces whichever of C/D survived, weighted by each one's own chance of
  // being there.
  const aChamp = 0.6 * (0.3 * winProb(0, 2) + 0.7 * winProb(0, 3));
  close(p[0][1], aChamp, 1e-9, 'A champion');

  const bChamp = 0.4 * (0.3 * winProb(1, 2) + 0.7 * winProb(1, 3));
  close(p[1][1], bChamp, 1e-9, 'B champion');

  const cChamp = 0.3 * (0.6 * winProb(2, 0) + 0.4 * winProb(2, 1));
  close(p[2][1], cChamp, 1e-9, 'C champion');

  const dChamp = 0.7 * (0.6 * winProb(3, 0) + 0.4 * winProb(3, 1));
  close(p[3][1], dChamp, 1e-9, 'D champion');
});

check('every round is a valid probability distribution over its winners', () => {
  // A 16-team bracket with an arbitrary but well-formed (antisymmetric) win
  // function. Every round must sum to exactly half the entrants into it, and
  // the last round -- one champion -- must sum to exactly 1.
  const n = 16;
  // Deterministic pseudo-random win function, antisymmetric by construction.
  const strength = Array.from({ length: n }, (_, i) => Math.sin(i * 12.9898) * 0.5 + 0.5);
  const winProb = (a, b) => {
    const sa = strength[a], sb = strength[b];
    const raw = 0.5 + 0.4 * (sa - sb);
    return Math.min(0.99, Math.max(0.01, raw));
  };
  // Antisymmetry is required by the function's own contract; verify the test
  // fixture actually has it before trusting the sums it implies.
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      if (a === b) continue;
      close(winProb(a, b) + winProb(b, a), 1, 1e-9, `winProb antisymmetry ${a},${b}`);
    }
  }

  const order = Array.from({ length: n }, (_, i) => i);
  const p = F.bracketAdvancementProbs(order, winProb);
  const rounds = Math.log2(n);

  for (let r = 0; r < rounds; r++) {
    const expected = n / Math.pow(2, r + 1);
    const sum = order.reduce((s, t) => s + p[t][r], 0);
    close(sum, expected, 1e-9, `round ${r} sums to ${expected} survivor(s)`);
  }
});

check('probability is monotone non-increasing round over round for every team', () => {
  const n = 8;
  const strength = [0.9, 0.1, 0.6, 0.4, 0.7, 0.3, 0.55, 0.45];
  const winProb = (a, b) => {
    const raw = 0.5 + 0.5 * (strength[a] - strength[b]);
    return Math.min(0.98, Math.max(0.02, raw));
  };
  const order = [0, 1, 2, 3, 4, 5, 6, 7];
  const p = F.bracketAdvancementProbs(order, winProb);
  for (const t of order) {
    for (let r = 1; r < p[t].length; r++) {
      assert.ok(p[t][r] <= p[t][r - 1] + 1e-12, `team ${t} round ${r} (${p[t][r]}) > round ${r - 1} (${p[t][r - 1]})`);
    }
  }
});

check('rejects a bracket size that is not a power of two', () => {
  assert.throws(() => F.bracketAdvancementProbs([0, 1, 2], () => 0.5), /power of two/);
});

console.log(`\n${passed} checks passed`);
