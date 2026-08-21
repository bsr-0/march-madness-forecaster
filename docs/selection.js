/* Bracket selection — a MIRROR of src/product/selection.py.
 *
 * Python is canonical. This file must not invent selection logic, add an
 * objective, or reinterpret the artifact. tests/test_product_parity.py
 * generates fixtures from the Python implementation and asserts this file
 * reproduces the selected candidate indices exactly.
 *
 * If something here is awkward to express, the fix is to make the artifact
 * contract clearer on the Python side — not to let the two implementations
 * drift apart.
 */

// Index into a candidate's per-round winner lists (w).
const R32 = 0, S16 = 1, E8 = 2, F4 = 3, FINAL = 4, CHAMP = 5;

// Frozen v1 objectives. No blend, no ownership penalty.
const OBJECTIVES = ['ev', 'p1'];

/* Frozen preference predicates, keyed as in the artifact. */
function preferencePredicates(artifact) {
  const seedOf = i => artifact.teams[i].seed;
  const seedsIn = (w, rnd) => w[rnd].map(seedOf);
  return {
    none: () => true,
    f4_at_least_1_two_three: w => seedsIn(w, F4).filter(s => s === 2 || s === 3).length >= 1,
    f4_at_least_2_two_three: w => seedsIn(w, F4).filter(s => s === 2 || s === 3).length >= 2,
    f4_mostly_favorites:     w => seedsIn(w, F4).filter(s => s === 1).length >= 3,
    s16_at_least_1_double_digit: w => seedsIn(w, S16).some(s => s >= 10),
    s16_at_least_2_double_digit: w => seedsIn(w, S16).filter(s => s >= 10).length >= 2,
    s16_no_double_digit:         w => !seedsIn(w, S16).some(s => s >= 10),
  };
}

/* The one parameterised predicate. */
function teamReachesFinalFour(teamIndex) {
  return w => w[F4].includes(teamIndex);
}

/* Return k candidate indices: highest-scoring, distinct compositions.
 *
 * Hierarchical diversity exactly as frozen — distinct champion, then distinct
 * Final Four, then top up. Plain top-k returns k one-pick variations of one
 * bracket, which is the collapse this exists to prevent.
 *
 * Ties break by candidate index so the result is deterministic and does not
 * depend on Array.prototype.sort stability.
 */
function selectBrackets(artifact, objective = 'ev', preference = 'none', teamIndex = null, k = 3) {
  if (!OBJECTIVES.includes(objective)) throw new Error(`unknown objective ${objective}`);

  const preds = preferencePredicates(artifact);
  let pred;
  if (preference === 'team_reaches_final_four') {
    if (teamIndex == null) throw new Error('team_reaches_final_four requires teamIndex');
    pred = teamReachesFinalFour(teamIndex);
  } else if (preds[preference]) {
    pred = preds[preference];
  } else {
    throw new Error(`unknown preference ${preference}`);
  }

  const C = artifact.candidates;
  const surviving = [];
  for (let i = 0; i < C.length; i++) if (pred(C[i].w)) surviving.push(i);
  surviving.sort((a, b) => (C[b][objective] - C[a][objective]) || (a - b));

  const chosen = [], usedChamps = new Set(), usedF4 = new Set();
  const f4key = i => C[i].w[F4].slice().sort((x, y) => x - y).join(',');

  for (const i of surviving) {               // tier 1 — distinct champion
    if (chosen.length >= k) break;
    const champ = C[i].w[CHAMP][0];
    if (!usedChamps.has(champ)) {
      chosen.push(i); usedChamps.add(champ); usedF4.add(f4key(i));
    }
  }
  for (const i of surviving) {               // tier 2 — distinct Final Four
    if (chosen.length >= k) break;
    const f = f4key(i);
    if (!usedF4.has(f) && !chosen.includes(i)) { chosen.push(i); usedF4.add(f); }
  }
  for (const i of surviving) {               // tier 3 — top up
    if (chosen.length >= k) break;
    if (!chosen.includes(i)) chosen.push(i);
  }
  return chosen;
}

/* User-facing "happens in X of 10 tournaments".
 *
 * Read from the artifact's full-bank fields, NEVER by counting candidates. The
 * sampler deliberately over-samples unlikely champions to protect diversity, so
 * counting the candidate list would bias every frequency toward rare scenarios.
 */
function constraintFrequency(artifact, preference, teamId = '') {
  if (preference === 'none') return 1.0;
  if (preference === 'team_reaches_final_four') {
    return (artifact.team_final_four_probabilities || {})[teamId] ?? null;
  }
  return (artifact.constraint_probabilities || {})[preference] ?? null;
}

/* Champion, Final Four and upset profile for one candidate. */
function candidateSummary(artifact, index) {
  const c = artifact.candidates[index], teams = artifact.teams;
  const champ = c.w[CHAMP][0];
  return {
    index,
    champion_id:   teams[champ].id,
    champion_seed: teams[champ].seed,
    final_four: c.w[F4]
      .map(t => ({ id: teams[t].id, seed: teams[t].seed }))
      .sort((a, b) => (a.seed - b.seed) || a.id.localeCompare(b.id)),
    double_digit_s16: c.dd16,
    ev: c.ev,
    p1: c.p1,
  };
}

/* Plain-language differences against a baseline bracket.
 *
 * Built only from candidate metadata already in the artifact. Deliberately makes
 * no claim about which bracket is better — the two objectives are near
 * orthogonal, so "different" is the honest framing.
 */
function whyThisDiffers(artifact, index, baselineIndex) {
  const a = candidateSummary(artifact, index), b = candidateSummary(artifact, baselineIndex);
  const out = [];
  if (a.champion_id !== b.champion_id) {
    out.push(`Takes ${a.champion_id} (${a.champion_seed}) as champion instead of ` +
             `${b.champion_id} (${b.champion_seed}).`);
  }
  const aF4 = new Set(a.final_four.map(t => t.id)), bF4 = new Set(b.final_four.map(t => t.id));
  const added = [...aF4].filter(x => !bF4.has(x)).sort();
  const dropped = [...bF4].filter(x => !aF4.has(x)).sort();
  if (added.length || dropped.length) {
    const bits = [];
    if (added.length) bits.push('adds ' + added.join(', '));
    if (dropped.length) bits.push('drops ' + dropped.join(', '));
    out.push('Final Four ' + bits.join(' and ') + '.');
  }
  if (a.double_digit_s16 !== b.double_digit_s16) {
    out.push(`Advances ${a.double_digit_s16} double-digit seed(s) to the Sweet 16 ` +
             `rather than ${b.double_digit_s16}.`);
  }
  if (!out.length) out.push('Differs only in individual game picks, not in overall shape.');
  return out;
}

/* Expand a candidate into the shape the existing bracket renderer expects.
 *
 * Reuses the renderer's contract rather than introducing a second bracket
 * representation. Display win_prob is computed browser-side from team ratings,
 * exactly as the existing chalk path does — that is presentation, which the
 * browser owns.
 */
function candidateToRounds(artifact, index, mkTeamFn, log5Fn) {
  const ROUND_NAMES = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'];
  const teams = artifact.teams;
  const c = artifact.candidates[index];
  let current = artifact.first_round.slice();
  const rounds = [];

  for (let r = 0; r < 6; r++) {
    const winners = new Set(c.w[r]);
    const games = [];
    const next = [];
    for (let g = 0; g < current.length; g += 2) {
      const i1 = current[g], i2 = current[g + 1];
      const t1 = mkTeamFn(teams[i1].id, teams[i1].id, teams[i1].seed, 0.5);
      const t2 = mkTeamFn(teams[i2].id, teams[i2].id, teams[i2].seed, 0.5);
      const winnerIdx = winners.has(i1) ? i1 : i2;
      const wp = log5Fn(t1.barthag, t2.barthag);
      games.push({
        round: ROUND_NAMES[r],
        region: teams[i1].region === teams[i2].region ? teams[i1].region : '',
        team1: t1, team2: t2,
        win_prob: wp,
        is_upset: t1.seed !== t2.seed &&
          (winnerIdx === i1 ? t1.seed > t2.seed : t2.seed > t1.seed),
        precomputed_winner_id: teams[winnerIdx].id,
        team1_pool_pct: null, team2_pool_pct: null,
      });
      next.push(winnerIdx);
    }
    rounds.push({ round_name: ROUND_NAMES[r], games });
    current = next;
  }
  return rounds;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    selectBrackets, constraintFrequency, candidateSummary, whyThisDiffers,
    candidateToRounds, preferencePredicates, OBJECTIVES,
  };
}
