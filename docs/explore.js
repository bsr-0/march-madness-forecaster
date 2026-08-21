/* EXPLORE — what the model thinks is likely.
 *
 * Read-only. There are no bracket-generation controls here: building is Build's
 * job, and duplicating it would recreate the "same application, rearranged"
 * information architecture this destination replaces.
 *
 * Every number comes from the artifact's `team_round_probabilities`, counted
 * over the FULL simulated bank — never over `candidates`, which deliberately
 * over-samples unlikely champions to protect bracket diversity and would
 * overstate exactly the long shots a reader is most likely to misread.
 *
 * No probability is computed in this file. Same rule as the bracket board: the
 * artifact is the contract.
 */

// Column indices into a team's advancement row.
const XP_R32 = 0, XP_S16 = 1, XP_E8 = 2, XP_F4 = 3, XP_FINAL = 4, XP_CHAMP = 5;

const XP_STAGES = [
  { col: XP_R32,   label: 'Round of 32' },
  { col: XP_S16,   label: 'Sweet 16' },
  { col: XP_E8,    label: 'Elite 8' },
  { col: XP_F4,    label: 'Final Four' },
  { col: XP_FINAL, label: 'Final' },
  { col: XP_CHAMP, label: 'Champion' },
];

let exploreState = { sort: XP_CHAMP, query: '' };

function initExplore() {
  const el = document.getElementById('explore-headline');
  if (!ARTIFACT || !ARTIFACT.team_round_probabilities) {
    // Build owns the fetch; if the user reaches Explore first, retry shortly
    // rather than rendering an empty page.
    if (el) el.innerHTML = '<p class="empty-note">Loading…</p>';
    setTimeout(() => { if (ARTIFACT) renderExplore(); }, 300);
    return;
  }
  renderExplore();
}

function xpPct(v) {
  if (v >= 0.995) return '>99%';
  if (v > 0 && v < 0.005) return '<1%';
  return `${Math.round(v * 100)}%`;
}

/* "In 10 tournaments, this happens about N times" — the framing a reader can
 * actually act on. Kept out of the table, where percentages read better. */
function xpInTen(v) {
  const n = v * 10;
  if (n >= 9.5) return 'nearly every time';
  if (n < 0.5) return 'rarely';
  return `about ${n < 1 ? n.toFixed(1) : Math.round(n)} times in 10`;
}

function exploreRows() {
  const trp = ARTIFACT.team_round_probabilities;
  const rows = ARTIFACT.teams.map((t, i) => ({ ...t, p: trp[i] }));
  const q = exploreState.query;
  const filtered = q ? rows.filter(r => r.name.toLowerCase().includes(q)) : rows;
  return filtered.sort((a, b) => (b.p[exploreState.sort] - a.p[exploreState.sort])
                              || a.id.localeCompare(b.id));
}

function setExploreSort(col) { exploreState.sort = Number(col); renderExplore(); }
function setExploreSearch(q) { exploreState.query = q.trim().toLowerCase(); renderExplore(); }

function renderExplore() {
  const trp = ARTIFACT.team_round_probabilities;
  const teams = ARTIFACT.teams;

  const ranked = col => teams.map((t, i) => ({ ...t, v: trp[i][col] }))
    .sort((a, b) => b.v - a.v || a.id.localeCompare(b.id));

  // Headline: who wins it, and who gets to the last weekend.
  const champs = ranked(XP_CHAMP).slice(0, 6);
  const f4 = ranked(XP_F4).slice(0, 6);

  const bar = (v, max) => `<span class="xp-bar" style="width:${Math.max(2, (v / max) * 100)}%"></span>`;
  const cmax = champs[0].v, fmax = f4[0].v;

  document.getElementById('explore-headline').innerHTML = `
    <div class="xp-col">
      <p class="xp-col-title">Most likely champion</p>
      ${champs.map(t => `
        <div class="xp-row">
          <span class="xp-team">${t.name} <span class="xp-seed">(${t.seed})</span></span>
          <span class="xp-track">${bar(t.v, cmax)}</span>
          <span class="xp-val">${xpPct(t.v)}</span>
        </div>`).join('')}
      <p class="xp-note">No team is close to a favourite here — ${champs[0].name}
         wins ${xpInTen(champs[0].v)}, so most of the time the title goes elsewhere.</p>
    </div>
    <div class="xp-col">
      <p class="xp-col-title">Most likely to reach the Final Four</p>
      ${f4.map(t => `
        <div class="xp-row">
          <span class="xp-team">${t.name} <span class="xp-seed">(${t.seed})</span></span>
          <span class="xp-track">${bar(t.v, fmax)}</span>
          <span class="xp-val">${xpPct(t.v)}</span>
        </div>`).join('')}
      <p class="xp-note">Four teams reach the Final Four, so these add up to four
         across the whole field — not to one.</p>
    </div>`;

  // Full advancement table: every team, every stage.
  const rows = exploreRows();
  document.getElementById('explore-table').innerHTML = `
    <table class="xp-table">
      <thead><tr>
        <th class="xp-th-team">Team</th>
        <th>Seed</th>
        ${XP_STAGES.map(s => `
          <th class="xp-th-sortable${exploreState.sort === s.col ? ' sorted' : ''}"
              onclick="setExploreSort(${s.col})">${s.label}</th>`).join('')}
      </tr></thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td class="xp-td-team">${r.name}</td>
            <td class="xp-td-seed">${r.seed}</td>
            ${XP_STAGES.map(s => `
              <td class="xp-td-p${r.p[s.col] >= 0.5 ? ' strong' : ''}">${xpPct(r.p[s.col])}</td>`).join('')}
          </tr>`).join('')}
      </tbody>
    </table>
    ${rows.length ? '' : '<p class="empty-note">No teams match that search.</p>'}`;

  renderExploreMatchups();
}

/* Opening-round matchups, closest games first.
 *
 * The pairings come from the artifact's `first_round` bracket order and the
 * probabilities from its `pairwise` table, so this agrees with the bracket board
 * by construction rather than by coincidence.
 */
function renderExploreMatchups() {
  const teams = ARTIFACT.teams;
  const order = ARTIFACT.first_round;
  const games = [];
  for (let g = 0; g < order.length; g += 2) {
    const i1 = order[g], i2 = order[g + 1];
    const p = pairwiseProb(ARTIFACT, i1, i2);
    const favFirst = p >= 0.5;
    games.push({
      fav: teams[favFirst ? i1 : i2],
      dog: teams[favFirst ? i2 : i1],
      p: favFirst ? p : 1 - p,
    });
  }
  const closest = games.sort((a, b) => a.p - b.p).slice(0, 8);

  document.getElementById('explore-matchups').innerHTML = closest.map(g => `
    <div class="xp-match">
      <span class="xp-match-teams">
        ${g.fav.name} <span class="xp-seed">(${g.fav.seed})</span>
        <span class="xp-vs">vs</span>
        ${g.dog.name} <span class="xp-seed">(${g.dog.seed})</span>
      </span>
      <span class="xp-match-p">${xpPct(g.p)}</span>
    </div>`).join('');
}
