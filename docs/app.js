/* ── March Madness Forecaster — App ── */

// ── State ──
let bracketData = null;
let validationData = null;
let dashboardData = null;
let modelMetrics = null;
let currentRound = 'Round of 64';
let currentRegion = 'All';

// ── Boot ──
document.addEventListener('DOMContentLoaded', async () => {
  const [b, v, d, m] = await Promise.all([
    fetch('data/bracket_2026.json').then(r => r.json()),
    fetch('data/validation_2025.json').then(r => r.json()),
    fetch('data/dashboard.json').then(r => r.json()),
    fetch('data/model_metrics.json').then(r => r.json()),
  ]);
  bracketData = b;
  validationData = v;
  dashboardData = d;
  modelMetrics = m;

  renderPredictions();
  renderBacktest();
});

// ── Tab Switching ──
function switchTab(tab) {
  document.getElementById('section-predictions').classList.toggle('hidden', tab !== 'predictions');
  document.getElementById('section-backtest').classList.toggle('hidden', tab !== 'backtest');
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
// SECTION 1: 2026 PREDICTIONS
// ═══════════════════════════════════════════════════════════════════

function renderPredictions() {
  renderChampionshipProbs();
  renderRoundSelector();
  renderRegionFilter();
  renderBracketGames();
  renderUpsetAlerts();
}

// ── Championship Probability Cards ──
function renderChampionshipProbs() {
  const container = document.getElementById('champ-probs');
  const teams = bracketData.championship_probabilities.slice(0, 16);
  const maxProb = teams[0]?.championship_prob || 0.1;

  container.innerHTML = teams.map((t, i) => {
    const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
    const pct = (t.championship_prob * 100).toFixed(1);
    const barW = ((t.championship_prob / maxProb) * 100).toFixed(0);
    const ff = (t.final_four_prob * 100).toFixed(1);
    const e8 = (t.elite_eight_prob * 100).toFixed(1);
    return `
      <div class="champ-card">
        <div class="champ-rank ${rankClass}">${i + 1}</div>
        <div class="champ-info">
          <div class="champ-name">${t.team_name}</div>
          <div class="champ-details">(${t.seed}) ${t.region} · ${t.conference}</div>
          <div class="champ-details">F4 ${ff}% · E8 ${e8}%</div>
          <div class="champ-bar-bg"><div class="champ-bar-fill" style="width:${barW}%"></div></div>
        </div>
        <div class="champ-prob">${pct}%</div>
      </div>`;
  }).join('');
}

// ── Round Selector ──
function renderRoundSelector() {
  const container = document.getElementById('round-selector');
  const rounds = bracketData.rounds.map(r => r.round_name);
  container.innerHTML = rounds.map(r => {
    const cls = r === currentRound ? 'active' : '';
    return `<button class="round-btn ${cls}" onclick="selectRound('${r}')">${r}</button>`;
  }).join('');
}

function selectRound(round) {
  currentRound = round;
  renderRoundSelector();
  renderRegionFilter();
  renderBracketGames();
}

// ── Region Filter ──
function renderRegionFilter() {
  const container = document.getElementById('region-filter');
  const roundObj = bracketData.rounds.find(r => r.round_name === currentRound);
  if (!roundObj) return;

  const regions = ['All', ...new Set(roundObj.games.map(g => g.region))];
  container.innerHTML = regions.map(r => {
    const cls = r === currentRegion ? 'active' : '';
    return `<button class="region-btn ${cls}" onclick="selectRegion('${r}')">${r}</button>`;
  }).join('');
}

function selectRegion(region) {
  currentRegion = region;
  renderRegionFilter();
  renderBracketGames();
}

// ── Bracket Game Cards ──
function renderBracketGames() {
  const container = document.getElementById('bracket-games');
  const roundObj = bracketData.rounds.find(r => r.round_name === currentRound);
  if (!roundObj) { container.innerHTML = ''; return; }

  let games = roundObj.games;
  if (currentRegion !== 'All') {
    games = games.filter(g => g.region === currentRegion);
  }

  container.innerHTML = games.map(g => gameCardHTML(g)).join('');
}

function gameCardHTML(g) {
  const t1Win = g.winner_id === g.team1_id;
  const prob = (g.win_prob * 100).toFixed(1);
  const probClass = g.win_prob >= 0.75 ? 'high' : g.win_prob >= 0.55 ? 'mid' : 'low';
  const upsetClass = g.is_upset ? 'upset' : '';

  const t1Class = t1Win ? 'winner' : 'loser';
  const t2Class = t1Win ? 'loser' : 'winner';
  const t1Prob = t1Win ? prob : (100 - parseFloat(prob)).toFixed(1);
  const t2Prob = t1Win ? (100 - parseFloat(prob)).toFixed(1) : prob;
  const t1ProbClass = parseFloat(t1Prob) >= 75 ? 'high' : parseFloat(t1Prob) >= 55 ? 'mid' : 'low';
  const t2ProbClass = parseFloat(t2Prob) >= 75 ? 'high' : parseFloat(t2Prob) >= 55 ? 'mid' : 'low';

  return `
    <div class="game-card ${upsetClass}">
      <div class="game-team ${t1Class}">
        <div class="flex items-center">
          <span class="team-seed ${seedClass(g.team1_seed)}">${g.team1_seed}</span>
          <span class="team-name">${g.team1.replace(/^\(\d+\)\s*/, '')}</span>
        </div>
        <span class="win-prob ${t1ProbClass}">${t1Prob}%</span>
      </div>
      <div class="game-team ${t2Class}">
        <div class="flex items-center">
          <span class="team-seed ${seedClass(g.team2_seed)}">${g.team2_seed}</span>
          <span class="team-name">${g.team2.replace(/^\(\d+\)\s*/, '')}</span>
        </div>
        <span class="win-prob ${t2ProbClass}">${t2Prob}%</span>
      </div>
      <div class="game-meta">
        <span>${g.region}</span>
        <span>${g.round}</span>
        ${g.is_upset ? '<span class="upset-badge">Upset</span>' : ''}
      </div>
    </div>`;
}

function seedClass(seed) {
  if (seed <= 4) return 'top-seed';
  if (seed <= 8) return 'mid-seed';
  return 'low-seed';
}

// ── Upset Alerts ──
function renderUpsetAlerts() {
  const container = document.getElementById('upset-alerts');
  const upsets = [];
  bracketData.rounds.forEach(r => {
    r.games.forEach(g => {
      if (g.is_upset) upsets.push(g);
    });
  });
  if (upsets.length === 0) {
    container.innerHTML = '<p class="text-gray-500">No upsets predicted.</p>';
    return;
  }
  container.innerHTML = upsets.map(g => gameCardHTML(g)).join('');
}

// ═══════════════════════════════════════════════════════════════════
// SECTION 2: 2025 BACKTEST VALIDATION
// ═══════════════════════════════════════════════════════════════════

function renderBacktest() {
  renderHeadlineMetrics();
  renderKaggleMetrics();
  renderESPNMetrics();
  renderCalibrationChart();
  renderPerYearTable();
  renderRoundAccuracyChart();
  renderIntegrityNote();
}

// ── Headline Metrics ──
function renderHeadlineMetrics() {
  const container = document.getElementById('headline-metrics');
  const o = validationData.overall;
  const bt = dashboardData.backtest;

  const metrics = [
    { value: (o.accuracy * 100).toFixed(1) + '%', label: 'Overall Accuracy', color: 'text-green-400' },
    { value: o.brier_score.toFixed(4), label: 'Brier Score', color: 'text-blue-400' },
    { value: o.log_loss.toFixed(4), label: 'Log Loss', color: 'text-purple-400' },
    { value: o.n_games, label: 'Games Tested', color: 'text-indigo-400' },
    { value: bt.kaggle?.estimated_rank || 'N/A', label: 'Est. Kaggle Rank', color: 'text-yellow-400' },
    { value: bt.espn_pool ? '#' + bt.espn_pool.rank_position + '/' + bt.espn_pool.pool_size : 'N/A', label: 'ESPN Pool Rank', color: 'text-orange-400' },
  ];

  container.innerHTML = metrics.map(m => `
    <div class="metric-card">
      <div class="metric-value ${m.color}">${m.value}</div>
      <div class="metric-label">${m.label}</div>
    </div>`).join('');
}

// ── Kaggle Metrics ──
function renderKaggleMetrics() {
  const container = document.getElementById('kaggle-metrics');
  const k = dashboardData.backtest.kaggle;
  if (!k) { container.innerHTML = '<p class="text-gray-500">No Kaggle data available.</p>'; return; }

  const rows = [
    { label: 'Estimated Rank', value: k.estimated_rank, cls: 'great' },
    { label: 'Brier Score', value: k.brier?.toFixed(4), cls: 'good' },
    { label: 'Round-Weighted Brier', value: k.round_weighted_brier?.toFixed(4), cls: '' },
    { label: 'Accuracy', value: (k.accuracy * 100).toFixed(1) + '%', cls: 'good' },
  ];

  container.innerHTML = rows.map(r => `
    <div class="stat-row">
      <span class="stat-label">${r.label}</span>
      <span class="stat-value ${r.cls}">${r.value}</span>
    </div>`).join('');
}

// ── ESPN Pool Metrics ──
function renderESPNMetrics() {
  const container = document.getElementById('espn-metrics');
  const e = dashboardData.backtest.espn_pool;
  if (!e) { container.innerHTML = '<p class="text-gray-500">No ESPN pool data available.</p>'; return; }

  const percentile = ((1 - e.rank_position / e.pool_size) * 100).toFixed(0);
  const rows = [
    { label: 'Pool Size', value: e.pool_size, cls: '' },
    { label: 'Rank Position', value: '#' + e.rank_position, cls: 'good' },
    { label: 'Percentile', value: percentile + 'th', cls: 'great' },
    { label: 'Total Score', value: e.score.toLocaleString(), cls: 'good' },
  ];

  container.innerHTML = rows.map(r => `
    <div class="stat-row">
      <span class="stat-label">${r.label}</span>
      <span class="stat-value ${r.cls}">${r.value}</span>
    </div>`).join('');
}

// ── Calibration Chart ──
function renderCalibrationChart() {
  const ctx = document.getElementById('calibration-chart').getContext('2d');
  const cal = validationData.calibration;
  const labels = cal.map(c => c.bin_center);
  const predicted = cal.map(c => c.predicted_avg);
  const actual = cal.map(c => c.actual_avg);
  const counts = cal.map(c => c.count);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.map(l => (l * 100).toFixed(0) + '%'),
      datasets: [
        {
          label: 'Perfect Calibration',
          data: labels,
          borderColor: '#4b5563',
          borderDash: [6, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
        {
          label: 'Predicted Avg',
          data: predicted,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.15)',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: '#6366f1',
          fill: false,
        },
        {
          label: 'Actual Win Rate',
          data: actual,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.15)',
          borderWidth: 2,
          pointRadius: 5,
          pointBackgroundColor: '#f59e0b',
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: { display: true, text: 'Predicted Probability Bin', color: '#9ca3af' },
          ticks: { color: '#6b7280' },
          grid: { color: '#1f2937' },
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'Frequency / Probability', color: '#9ca3af' },
          ticks: { color: '#6b7280', callback: v => (v * 100) + '%' },
          grid: { color: '#1f2937' },
        },
      },
      plugins: {
        legend: { labels: { color: '#d1d5db' } },
        tooltip: {
          callbacks: {
            afterLabel: function(ctx) {
              if (ctx.datasetIndex === 2) {
                return 'n = ' + counts[ctx.dataIndex];
              }
            }
          }
        }
      },
    },
  });
}

// ── Per-Year Table ──
function renderPerYearTable() {
  const tbody = document.getElementById('per-year-body');
  const years = validationData.per_year;

  tbody.innerHTML = years.map(y => {
    const skillColor = y.skill_score >= 0 ? 'text-green-400' : 'text-red-400';
    const skillSign = y.skill_score >= 0 ? '+' : '';
    return `
      <tr>
        <td class="font-bold text-white">${y.year}</td>
        <td>${y.n_games}</td>
        <td class="text-green-400 font-semibold">${(y.accuracy * 100).toFixed(1)}%</td>
        <td>${y.brier_score.toFixed(4)}</td>
        <td class="text-gray-500">${y.seed_baseline_brier.toFixed(4)}</td>
        <td class="${skillColor} font-semibold">${skillSign}${(y.skill_score * 100).toFixed(1)}%</td>
        <td>${y.upsets_correctly_predicted}/${y.upsets_actual}</td>
      </tr>`;
  }).join('');
}

// ── Round Accuracy Chart ──
function renderRoundAccuracyChart() {
  const canvas = document.getElementById('round-accuracy-chart');
  const ctx = canvas.getContext('2d');

  // Try model_metrics first, then aggregate from per-year round_breakdowns
  let ra = modelMetrics.round_accuracy || {};
  if (Object.keys(ra).length === 0 && validationData.per_year) {
    const agg = {};
    validationData.per_year.forEach(y => {
      const rb = y.round_breakdown || {};
      Object.entries(rb).forEach(([round, stats]) => {
        if (!agg[round]) agg[round] = { correct: 0, total: 0 };
        agg[round].correct += stats.correct || 0;
        agg[round].total += stats.total || 0;
      });
    });
    Object.entries(agg).forEach(([round, s]) => {
      if (s.total > 0) ra[round] = { accuracy: s.correct / s.total, total: s.total };
    });
  }

  const labels = Object.keys(ra);
  if (labels.length === 0) {
    canvas.parentElement.innerHTML = '<p class="text-gray-500 text-center py-8">No round-level accuracy data available yet. Run the backtest pipeline to generate this data.</p>';
    return;
  }
  const accs = labels.map(l => ra[l].accuracy);
  const totals = labels.map(l => ra[l].total);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Accuracy',
        data: accs,
        backgroundColor: accs.map(a => a >= 0.75 ? 'rgba(52,211,153,0.7)' : a >= 0.65 ? 'rgba(251,191,36,0.7)' : 'rgba(248,113,113,0.7)'),
        borderColor: accs.map(a => a >= 0.75 ? '#34d399' : a >= 0.65 ? '#fbbf24' : '#f87171'),
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0, max: 1,
          ticks: { color: '#6b7280', callback: v => (v * 100) + '%' },
          grid: { color: '#1f2937' },
          title: { display: true, text: 'Accuracy', color: '#9ca3af' },
        },
        x: {
          ticks: { color: '#d1d5db' },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const idx = ctx.dataIndex;
              return `${(ctx.raw * 100).toFixed(1)}% (${totals[idx]} games)`;
            }
          }
        }
      },
    },
  });
}

// ── Integrity Note ──
function renderIntegrityNote() {
  const el = document.getElementById('integrity-note');
  const integrity = dashboardData.backtest.integrity;
  if (integrity) {
    el.innerHTML = `
      <strong>Level ${integrity.level}:</strong> ${integrity.note}<br>
      <span class="text-gray-500 text-xs mt-1 inline-block">Source: ${integrity.source}</span>
    `;
  } else {
    el.textContent = 'No integrity metadata available. Backtest results should be interpreted as retrospective.';
  }
}
