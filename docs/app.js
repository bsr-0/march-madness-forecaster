const DEFAULT_URL = 'data/dashboard.json';

const metaEl = document.getElementById('meta');
const headlineEl = document.getElementById('headline-metrics');
const trainingCardsEl = document.getElementById('training-cards');
const gamesChartEl = document.getElementById('games-chart');
const trainingNoteEl = document.getElementById('training-note');
const featureChartEl = document.getElementById('feature-chart');
const featureTableEl = document.getElementById('feature-table');
const featureNoteEl = document.getElementById('feature-note');
const matchupTableEl = document.getElementById('matchup-table');
const championTableEl = document.getElementById('champion-table');
const predictionNoteEl = document.getElementById('prediction-note');
const backtestCardsEl = document.getElementById('backtest-cards');
const backtestNoteEl = document.getElementById('backtest-note');

const fileInput = document.getElementById('file-input');
const reloadBtn = document.getElementById('reload');

const fmt = (value, digits = 2) => {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return value;
};

const pct = (value, digits = 1) => {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(digits)}%`;
};

const clearEl = (el) => {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
};

const addCard = (container, label, value, hint) => {
  const card = document.createElement('div');
  card.className = 'card';
  const labelEl = document.createElement('div');
  labelEl.className = 'label';
  labelEl.textContent = label;
  const valueEl = document.createElement('div');
  valueEl.className = 'value';
  valueEl.textContent = value;
  card.appendChild(labelEl);
  card.appendChild(valueEl);
  if (hint) {
    const hintEl = document.createElement('div');
    hintEl.className = 'label';
    hintEl.textContent = hint;
    card.appendChild(hintEl);
  }
  container.appendChild(card);
};

const renderMeta = (metadata) => {
  clearEl(metaEl);
  if (!metadata) return;
  const items = [
    metadata.model_name ? `Model: ${metadata.model_name}` : null,
    metadata.generated_at ? `Generated: ${metadata.generated_at}` : null,
    metadata.training_seasons ? `Training seasons: ${metadata.training_seasons.join(', ')}` : null,
    metadata.holdout_season ? `Holdout season: ${metadata.holdout_season}` : null,
  ].filter(Boolean);

  items.forEach((item) => {
    const span = document.createElement('span');
    span.textContent = item;
    metaEl.appendChild(span);
  });
};

const renderHeadline = (summary, predictions, backtest) => {
  clearEl(headlineEl);
  addCard(headlineEl, 'Training Games', fmt(summary?.total_games, 0));
  addCard(headlineEl, 'Teams', fmt(summary?.unique_teams, 0));
  addCard(headlineEl, 'Features', fmt(summary?.feature_count, 0));
  addCard(headlineEl, 'Holdout Brier', fmt(backtest?.metrics?.brier, 3), backtest?.status || '');
  addCard(headlineEl, 'Top Champion', predictions?.champion_probs?.[0]?.team || '—');
};

const renderTraining = (summary) => {
  clearEl(trainingCardsEl);
  if (!summary) return;
  addCard(trainingCardsEl, 'Seasons', summary.seasons?.length || 0, summary.seasons?.[0] ? `${summary.seasons[0]}-${summary.seasons.at(-1)}` : '');
  addCard(trainingCardsEl, 'Total Games', fmt(summary.total_games, 0));
  addCard(trainingCardsEl, 'Unique Teams', fmt(summary.unique_teams, 0));
  addCard(trainingCardsEl, 'Feature Count', fmt(summary.feature_count, 0));
  addCard(trainingCardsEl, 'Data Sources', summary.sources?.length || 0, summary.sources?.slice(0, 2).join(', '));

  clearEl(gamesChartEl);
  if (summary.games_per_season) {
    const max = Math.max(...Object.values(summary.games_per_season));
    Object.entries(summary.games_per_season).forEach(([season, games]) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      const label = document.createElement('span');
      label.textContent = season;
      const bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.width = `${(games / max) * 100}%`;
      const value = document.createElement('span');
      value.textContent = fmt(games, 0);
      row.appendChild(label);
      row.appendChild(bar);
      row.appendChild(value);
      gamesChartEl.appendChild(row);
    });
  }
  trainingNoteEl.textContent = summary.notes || '';
};

const renderFeatures = (featureImportance) => {
  clearEl(featureChartEl);
  clearEl(featureTableEl);
  if (!featureImportance?.features) return;
  const features = featureImportance.features.slice().sort((a, b) => b.importance - a.importance);
  const max = features[0]?.importance || 1;
  features.slice(0, 12).forEach((feature) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const label = document.createElement('span');
    label.textContent = feature.name.replace(/_/g, ' ');
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${(feature.importance / max) * 100}%`;
    const value = document.createElement('span');
    value.textContent = pct(feature.importance, 1);
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(value);
    featureChartEl.appendChild(row);
  });

  features.forEach((feature) => {
    const row = document.createElement('div');
    row.className = 'table-row';
    const name = document.createElement('span');
    name.textContent = feature.name.replace(/_/g, ' ');
    const value = document.createElement('span');
    value.textContent = pct(feature.importance, 1);
    row.appendChild(name);
    row.appendChild(value);
    featureTableEl.appendChild(row);
  });

  featureNoteEl.textContent = featureImportance.notes || '';
};

const renderPredictions = (predictions) => {
  clearEl(matchupTableEl);
  clearEl(championTableEl);
  if (!predictions) return;

  const matchups = predictions.matchups || [];
  matchups.slice(0, 16).forEach((match) => {
    const row = document.createElement('div');
    row.className = 'table-row';
    const label = document.createElement('span');
    label.textContent = `${match.region} · ${match.team1.seed} ${match.team1.name} vs ${match.team2.seed} ${match.team2.name}`;
    const value = document.createElement('span');
    value.textContent = pct(match.team1_win_prob, 1);
    row.appendChild(label);
    row.appendChild(value);
    matchupTableEl.appendChild(row);
  });

  (predictions.champion_probs || []).slice(0, 12).forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'table-row';
    const label = document.createElement('span');
    label.textContent = entry.team;
    const value = document.createElement('span');
    value.textContent = pct(entry.prob, 1);
    row.appendChild(label);
    row.appendChild(value);
    championTableEl.appendChild(row);
  });

  predictionNoteEl.textContent = predictions.notes || '';
};

const renderBacktest = (backtest) => {
  clearEl(backtestCardsEl);
  if (!backtest) return;
  addCard(backtestCardsEl, 'Holdout Season', backtest.season || '—', backtest.status || '');
  addCard(backtestCardsEl, 'Games', fmt(backtest.metrics?.games, 0));
  addCard(backtestCardsEl, 'Brier', fmt(backtest.metrics?.brier, 3));
  addCard(backtestCardsEl, 'Log Loss', fmt(backtest.metrics?.log_loss, 3));
  addCard(backtestCardsEl, 'Accuracy', pct(backtest.metrics?.accuracy, 1));
  addCard(backtestCardsEl, 'Brier Skill', pct(backtest.metrics?.brier_skill, 1));
  backtestNoteEl.textContent = backtest.notes || '';
};

const renderDashboard = (data) => {
  renderMeta(data.metadata);
  renderHeadline(data.training_summary, data.predictions, data.backtest);
  renderTraining(data.training_summary);
  renderFeatures(data.feature_importance);
  renderPredictions(data.predictions);
  renderBacktest(data.backtest);
};

const loadDefault = async () => {
  try {
    const res = await fetch(DEFAULT_URL, { cache: 'no-cache' });
    const data = await res.json();
    renderDashboard(data);
  } catch (err) {
    metaEl.textContent = 'Unable to load dashboard.json';
  }
};

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  const data = JSON.parse(text);
  renderDashboard(data);
});

reloadBtn.addEventListener('click', () => {
  fileInput.value = '';
  loadDefault();
});

loadDefault();
