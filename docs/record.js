/* TRACK RECORD — how the model has actually done.
 *
 * The integrity rule this file exists to honour: 2026 may never be presented as
 * prospective evidence. Spec 2027.v2 trains through it, so it is an in-sample
 * integration fixture.
 *
 * That separation is NOT enforced here. It is enforced in
 * scripts/generate_ml_backtest_data.py, which excludes the replay year from
 * every headline figure before the payload is written, and reports it apart in
 * `replay_year`. This file renders what it is given and never re-aggregates
 * per-year rows into a headline — doing so would silently reconstruct the
 * contaminated number the generator removed.
 *
 * For the record, the contamination was real and flattering: 2026 is the
 * model's best season (accuracy .746 vs .721 across the honest window, Brier
 * .145 vs .181), so including it improved every figure on this page.
 */

let RECORD = null;

async function initRecord() {
  try {
    const res = await fetch('data/ml_backtest.json?v=3');
    if (!res.ok) throw new Error(res.status);
    RECORD = await res.json();
  } catch (e) {
    document.getElementById('record-headline').innerHTML =
      '<p class="empty-note">Track record data is unavailable right now.</p>';
    return;
  }
  renderRecord();
}

function trPct(v) { return `${(v * 100).toFixed(1)}%`; }

function renderRecord() {
  const d = RECORD;
  const model = d.models.find(m => m.key === d.production_key);
  const seed = d.models.find(m => m.key === d.baseline_key);
  const years = d.years;

  /* Headline. Every figure here is out-of-sample by construction: the payload's
   * `years` excludes the replay year. */
  document.getElementById('record-headline').innerHTML = `
    <p class="tr-window">
      ${years[0]}–${years[years.length - 1]} · ${d.n_games.toLocaleString()} tournament games
    </p>
    <div class="tr-stats">
      <div class="tr-stat">
        <span class="tr-stat-val">${trPct(model.accuracy)}</span>
        <span class="tr-stat-lbl">of games called correctly</span>
        <span class="tr-stat-cmp">seed order alone: ${trPct(seed.accuracy)}</span>
      </div>
      <div class="tr-stat">
        <span class="tr-stat-val">${model.brier.toFixed(3)}</span>
        <span class="tr-stat-lbl">Brier score (lower is better)</span>
        <span class="tr-stat-cmp">seed order alone: ${seed.brier.toFixed(3)}</span>
      </div>
    </div>
    <p class="tr-plain">
      Picking every game by seed order gets you most of the way. The model's edge
      over that is real but modest, and it comes mainly from being better
      calibrated about how likely an upset is — not from calling many more games.
    </p>`;

  // Per-year, so a reader can see the variance rather than one flattering total.
  const py = d.per_year.filter(r => years.includes(r.year));
  const better = py.filter(r => r.brier_model < r.brier_seed).length;
  document.getElementById('record-per-year').innerHTML = `
    <p class="tr-section-title">Season by season</p>
    <p class="tr-sub">Beat seed order in ${better} of ${py.length} seasons.</p>
    <div class="tr-years">
      ${py.map(r => `
        <div class="tr-year${r.brier_model < r.brier_seed ? ' win' : ' loss'}">
          <span class="tr-year-y">${r.year}</span>
          <span class="tr-year-a">${trPct(r.accuracy_model)}</span>
        </div>`).join('')}
    </div>
    <p class="tr-note">Green means the model was better calibrated than seed order that season.</p>`;

  document.getElementById('record-method').innerHTML = renderRecordMethod();
  renderRecordReplay();
}

/* The replay, kept visually and semantically apart from the record above.
 * It is evidence that the pipeline runs end to end on a real field — not
 * evidence that the model predicts well. */
function renderRecordReplay() {
  const r = RECORD.replay_year;
  const el = document.getElementById('record-replay');
  if (!r) { el.innerHTML = ''; return; }

  el.innerHTML = `
    <div class="tr-replay">
      <p class="tr-replay-tag">Not part of the track record above</p>
      <p class="tr-section-title">${r.year}: a worked example, not a result</p>
      <p class="tr-plain">
        The ${r.year} tournament is what the Build tab uses to demonstrate itself.
        The model was built with that season already in hand, so it is not a
        prediction and its numbers say nothing about how the model would do on a
        season it has not seen. They are excluded from every figure above.
      </p>
      <p class="tr-replay-fig">
        For completeness: ${trPct(r.accuracy_model)} of ${r.n_games} games,
        against ${trPct(r.accuracy_seed)} for seed order.
        <span class="tr-replay-warn">Treat this as a self-test, not a score.</span>
      </p>
    </div>`;
}

/* How the model works, in the terms a reader actually needs. No algorithm names,
 * no optimiser vocabulary. */
function renderRecordMethod() {
  return `
    <p class="tr-section-title">How this works</p>
    <ul class="tr-method">
      <li>Every team gets a strength rating from its season, before the tournament starts.</li>
      <li>Those ratings give each possible game a win probability.</li>
      <li>The tournament is played out many thousands of times to see which brackets hold up.</li>
      <li>The method is fixed in advance and recorded, so it cannot be adjusted after results are known.</li>
    </ul>`;
}
