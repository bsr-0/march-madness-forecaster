/* Checks for the scorecard arithmetic in docs/fit.js (chalkRounds,
 * bracketScore, actualUpsets, modelVsSeed), on a hand-worked 4-team bracket
 * and on a real payload whose ESPN points are known independently.
 *
 * Run: node tests/test_scorecard.js
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const F = require(path.join(__dirname, '..', 'docs', 'fit.js'));

let failed = 0;
function check(name, fn) {
  try { fn(); console.log('  ok   ' + name); }
  catch (e) { failed++; console.error('  FAIL ' + name + '\n       ' + e.message); }
}

// Four teams, two rounds (the helpers walk six, so use a 64-slot tree whose
// other slots are dummies is overkill: the 6-round walk needs 64 slots).
// Build the smallest legal tree instead: 64 slots, seeds 1..16 repeated.
const first = Array.from({ length: 64 }, (_, i) => i);
const seeds = first.map(i => (i % 16) + 1);   // slot pairs (1,2), (3,4)... so the lower index is always the better seed

// A consistent reality: chalk, except that the teams in `flip` beat the better
// seed in their round-1 game, and everything downstream follows from that.
function play(flip) {
  let cur = first.slice();
  const out = [];
  for (let r = 0; r < 6; r++) {
    const next = [];
    for (let i = 0; i < cur.length; i += 2) {
      const a = cur[i], b = cur[i + 1];
      next.push(r === 0 && flip.includes(a) ? a : r === 0 && flip.includes(b) ? b : seeds[b] < seeds[a] ? b : a);
    }
    out.push(next); cur = next;
  }
  return out;
}

check('chalk takes the better seed every game, ties go to the earlier slot', () => {
  const rounds = F.chalkRounds(first, seeds);
  assert.strictEqual(rounds.length, 6);
  assert.deepStrictEqual(rounds.map(r => r.length), [32, 16, 8, 4, 2, 1]);
  for (const g of rounds[0]) assert.ok(seeds[g.win] <= seeds[g.win === g.a ? g.b : g.a]);
  // Slots 0 and 16 are both seed 1; whichever survives the tree is the earlier-slot one.
  assert.strictEqual(rounds[5][0].win, 0);
});

check('a chalk bracket scores every chalk game and nothing for an upset', () => {
  const chalk = F.chalkRounds(first, seeds);
  // Reality: the 2 seed in slot 1 beats the 1 seed in slot 0, who chalk then
  // also (wrongly) carries through every later round it reaches.
  const actual = play([1]);
  const s = F.bracketScore(chalk, actual, seeds);
  const carried = chalk.filter(r => r.some(g => g.win === 0)).length;   // rounds slot 0 is picked in
  assert.strictEqual(s.n, 63);
  assert.ok(carried >= 1);
  assert.ok(s.right <= 63 - carried);   // slot 0's own picks are all wrong; downstream picks may be too
  assert.strictEqual(s.rounds[0].right, 31);
  assert.strictEqual(s.upsetPicks, 0);
});

check('underdog picks are counted by the bracket\'s own matchup and paid by round', () => {
  const chalk = F.chalkRounds(first, seeds);
  const mine = chalk.map(r => r.map(g => ({ ...g })));
  mine[0][0].win = 1;   // pick the 2 seed over the 1 seed in round 1
  const s = F.bracketScore(mine, play([1]), seeds);   // reality agrees with the pick
  assert.strictEqual(s.upsetPicks, 1);
  assert.strictEqual(s.upsetRight, 1);
  assert.strictEqual(s.upsetPoints, 10);              // a round-1 underdog pays round-1 points
  assert.strictEqual(s.rounds[0].right, 32);
  assert.strictEqual(s.rounds[0].points, 320);
});

check('upset count on the real tree matches the winners', () => {
  assert.deepStrictEqual(F.actualUpsets(first, play([]), seeds), { upsets: 0, games: 63 });
  assert.strictEqual(F.actualUpsets(first, play([1]), seeds).upsets, 1);
  assert.strictEqual(F.actualUpsets(first, play([1, 3]), seeds).upsets, 2);
  assert.strictEqual(F.actualUpsets(first, [[], [], [], [], [], []], seeds), null);
});

check('modelVsSeed is paired: agreeing games cannot move the difference', () => {
  // Rows stored better-seed-first, so the seed is right when m > 0.
  const rows = [
    { m: 5, p: 3 },     // both right
    { m: -4, p: -1 },   // model right, seed wrong
    { m: -2, p: 6 },    // both wrong
    { m: 7, p: -2 },    // seed right, model wrong
    { m: 3, p: 1 },     // both right
    { m: 0, p: 4 },     // a push is not a decided game
  ];
  const v = F.modelVsSeed(rows);
  assert.strictEqual(v.n, 5);
  assert.strictEqual(v.model, 3 / 5);
  assert.strictEqual(v.seed, 3 / 5);
  assert.strictEqual(v.diff, 0);
  assert.strictEqual(v.modelOnly, 1);
  assert.strictEqual(v.seedOnly, 1);
  assert.strictEqual(F.modelVsSeed([]), null);
});

check('bracketScore reproduces the payload track_record, which was computed independently', () => {
  const dir = path.join(__dirname, '..', 'docs', 'data');
  let compared = 0;
  for (const y of [2023, 2024, 2025, 2026]) {
    const file = path.join(dir, `season_${y}.json`);
    if (!fs.existsSync(file)) continue;
    const s = JSON.parse(fs.readFileSync(file));
    if (!s.actual || !s.first_round || !s.track_record) continue;
    const sd = s.teams.map(t => t.seed);
    for (const st of s.strategies) {
      const rec = s.track_record.strategies[st.id];
      if (!rec) continue;
      const picks = st.picks.map(r => new Set(r));
      let cur = s.first_round.slice();
      const rounds = [];
      for (let r = 0; r < 6; r++) {
        const games = [], next = [];
        for (let i = 0; i < cur.length; i += 2) {
          const a = cur[i], b = cur[i + 1];
          const win = picks[r].has(a) ? a : b;
          games.push({ a, b, win }); next.push(win);
        }
        rounds.push(games); cur = next;
      }
      assert.strictEqual(F.bracketScore(rounds, s.actual, sd).points, rec.points, `${y} ${st.id}`);
      compared++;
    }
  }
  // Tolerates an absent data directory in CI, as the other data-backed checks do.
  if (!compared) console.log('       (no payloads with a track record found; skipped)');
});

if (failed) process.exitCode = 1;
