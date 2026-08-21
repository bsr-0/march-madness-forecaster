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

/* Diverse selection (product.v2) — mirror of src/product/selection.py.
 *
 * Versioned separately from the frozen 2027.v2 methodology: this changes only
 * WHICH already-scored candidates are shown, never how they were scored.
 *
 * v3 pins v2's semantics in configs/frozen/product_v3.json; the behaviour below
 * is unchanged from v2.
 *
 * v1 put "distinct champion" first, which on the 2026 field returned an
 * alternative retaining 0.973 EV with a Final Four identical to the baseline's,
 * while a 0.995 bracket with a changed Final Four went unshown. Champion
 * diversity is a signal, not a requirement.
 *
 * Not a distance metric. Hamming weights all 63 games equally, so R64 is 50.8%
 * of Hamming but 16.7% of the points — the wrong objective, as established.
 */
const SELECTION_VERSION = 'product.v3';
const MIN_F4_CHANGES = 1, MIN_S16_CHANGES = 2, DEFAULT_MIN_RETENTION = 0.97;
const DIVERSITY_TIERS = ['final_four', 'sweet_16', 'champion'];

function differenceProfile(artifact, index, baselineIndex) {
  const a = artifact.candidates[index].w, b = artifact.candidates[baselineIndex].w;
  const notIn = (xs, ys) => { const s = new Set(ys); return xs.filter(x => !s.has(x)).length; };
  return {
    champion: a[CHAMP][0] !== b[CHAMP][0] ? 1 : 0,
    final_four: notIn(a[F4], b[F4]),
    sweet_16: notIn(a[S16], b[S16]),
  };
}

function differsAtTier(artifact, index, otherIndex, tier) {
  const d = differenceProfile(artifact, index, otherIndex);
  if (tier === 'final_four') return d.final_four >= MIN_F4_CHANGES;
  if (tier === 'sweet_16')   return d.sweet_16 >= MIN_S16_CHANGES;
  if (tier === 'champion')   return d.champion >= 1;
  throw new Error(`unknown diversity tier ${tier}`);
}

/* A different champion is NOT sufficient on its own — that is the degenerate
 * case this exists to catch. */
function isMateriallyDifferent(artifact, index, baselineIndex) {
  return differsAtTier(artifact, index, baselineIndex, 'final_four') ||
         differsAtTier(artifact, index, baselineIndex, 'sweet_16');
}

/* Up to k brackets: quality first, subject to visible structural diversity.
 * Returns FEWER than k when the field has no distinguishable bracket left —
 * one honest bracket beats a manufactured second. */
function selectDiverse(artifact, objective = 'ev', k = 2, minRetention = DEFAULT_MIN_RETENTION) {
  if (!OBJECTIVES.includes(objective)) throw new Error(`unknown objective ${objective}`);
  if (k < 1) throw new Error('k must be at least 1');
  const C = artifact.candidates;
  if (!C.length) return [];

  const order = C.map((_, i) => i).sort((a, b) => (C[b][objective] - C[a][objective]) || (a - b));
  const chosen = [order[0]];
  const floor = C[order[0]][objective] * minRetention;

  while (chosen.length < k) {
    let pick = null;
    for (const tier of DIVERSITY_TIERS) {
      for (let n = 1; n < order.length; n++) {
        const i = order[n];
        if (C[i][objective] < floor) break;   // descending: nothing further qualifies
        if (chosen.includes(i)) continue;
        if (chosen.every(c => differsAtTier(artifact, i, c, tier))) { pick = i; break; }
      }
      if (pick !== null) break;
    }
    if (pick === null) break;
    chosen.push(pick);
  }
  return chosen;
}

/* The Build flow's selector: one bracket, plus an alternative if one exists. */
function selectWithAlternative(artifact, objective = 'ev', minRetention = DEFAULT_MIN_RETENTION) {
  return selectDiverse(artifact, objective, 2, minRetention);
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

/* ARTIFACT CONTRACT — mirror of src/product/artifact_contract.py.
 *
 * Ownership: the artifact contract owns the schema version, NOT the methodology
 * spec and NOT this selection code. Schema changes must be visible here rather
 * than discovered as a rendering bug.
 *
 * Strict in both directions. Refusing an older artifact is obvious; refusing a
 * NEWER one matters just as much, because it may carry fields this code does not
 * understand or may have redefined one it thinks it does. Rendering it anyway
 * would be a correctness failure that looks like a success.
 */
const EXPECTED_ARTIFACT_SCHEMA = 5;

const REQUIRED_ARTIFACT_FIELDS = [
  'schema', 'year', 'teams', 'first_round', 'pairwise',
  'candidates', 'team_round_probabilities', 'constraint_probabilities', 'meta',
];

function validateArtifact(artifact) {
  const declared = artifact ? artifact.schema : undefined;
  if (declared === undefined || declared === null) {
    throw new Error('artifact declares no schema; it predates the contract');
  }
  if (declared !== EXPECTED_ARTIFACT_SCHEMA) {
    throw new Error(
      `artifact is schema ${declared}, expected ${EXPECTED_ARTIFACT_SCHEMA}. ` +
      'Refusing rather than guessing.');
  }
  const missing = REQUIRED_ARTIFACT_FIELDS.filter(f => !(f in artifact) || !artifact[f]);
  if (missing.length) {
    throw new Error(`artifact is missing required field(s): ${missing.join(', ')}`);
  }
  const n = artifact.teams.length;
  if (artifact.pairwise.length !== n * n) {
    throw new Error(`pairwise has ${artifact.pairwise.length} entries, expected ${n * n}`);
  }
  if (artifact.team_round_probabilities.length !== n) {
    throw new Error('team_round_probabilities does not match the team table');
  }
  // Schema 5: canonical names are required. Rendering a slug is a product
  // defect, so it fails the contract rather than reaching a user.
  const nameless = artifact.teams.filter(t => !t.name);
  if (nameless.length) {
    throw new Error(`${nameless.length} team(s) have no canonical name`);
  }
  return true;
}

/* Read the canonical P(row beats col) out of the artifact's pairwise table.
 *
 * The browser does NOT compute this. Deriving it from ratings would make the
 * client a second, unversioned implementation of tournament math, and the board
 * could then disagree with the simulations that produced the bracket. The
 * artifact is the contract; if a number is needed for rendering, it ships.
 */
function pairwiseProb(artifact, i, j) {
  const n = artifact.teams.length;
  const p = artifact.pairwise;
  if (!p) throw new Error('artifact is missing the pairwise table (schema < 3)');
  return p[i * n + j];
}

/* Expand a candidate into the shape the existing bracket renderer expects.
 *
 * Reuses the renderer's contract rather than introducing a second bracket
 * representation.
 */
function candidateToRounds(artifact, index, mkTeamFn) {
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
      // Canonical names come from the artifact (schema 5); never derived here.
      const t1 = mkTeamFn(teams[i1].id, teams[i1].name, teams[i1].seed);
      const t2 = mkTeamFn(teams[i2].id, teams[i2].name, teams[i2].seed);
      const winnerIdx = winners.has(i1) ? i1 : i2;
      const wp = pairwiseProb(artifact, i1, i2);
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
    candidateToRounds, preferencePredicates, pairwiseProb, OBJECTIVES,
    validateArtifact, EXPECTED_ARTIFACT_SCHEMA,
    selectWithAlternative, selectDiverse, isMateriallyDifferent,
    differenceProfile, differsAtTier, SELECTION_VERSION,
  };
}
