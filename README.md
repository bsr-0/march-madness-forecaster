# March Madness Forecaster

NCAA Tournament prediction system that generates calibrated win probabilities and optimizes bracket picks.

## How it works

Six-phase pipeline:

1. **Data ingestion** — pulls historical game data, Torvik ratings, ESPN public picks, and Kaggle Massey Ordinals
2. **Feature engineering** — builds point-in-time team features (efficiency margins, Elo, SOS, momentum)
3. **Model training** — regularized logistic regression on regular-season games (2016–2024)
4. **Calibration** — temperature scaling on tournament games to correct probability distortion
5. **Simulation** — 50k Monte Carlo bracket simulations
6. **Optimization** — contrarian pick selection to maximize expected pool score

The production model is intentionally simple: 7 domain features, single logistic regression, no ensemble or neural components. Baseline experiments showed this matches or beats more complex approaches on tournament data.

## Usage

```bash
pip install -e .

# Full production run (2026)
march-madness run-production-2026

# Or run the general pipeline
march-madness ingest --year 2026
march-madness sota --year 2026
```

## Development

```bash
pytest           # run tests
ruff check src/  # lint
```

See `WORKFLOW.md` for the full pipeline diagram and design rationale.
