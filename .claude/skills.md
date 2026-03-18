# Skills — March Madness Forecaster

## Skill: Run Tests

Run the test suite before and after making changes.

```bash
# Fast unit tests
pytest tests/ -m "unit" -x --tb=short

# Full suite with coverage
pytest tests/ --cov=src --cov-report=term -x --tb=short

# Specific test file
pytest tests/test_<name>.py -v

# By marker
pytest tests/ -m "leakage"       # Data leakage tests
pytest tests/ -m "calibration"   # Calibration tests
pytest tests/ -m "production"    # Production path tests
```

Always run `pytest tests/ -m "unit" -x --tb=short` after code changes to catch regressions.

## Skill: Lint

```bash
ruff check src/ tests/
ruff check src/ tests/ --fix  # Auto-fix safe issues
```

Line length is 120. Target Python 3.9.

## Skill: Add a New Feature to the Pipeline

1. Add feature computation in `src/data/features/` — must be point-in-time safe (no future data leakage)
2. Update `TEAM_FEATURE_DIM` if the feature vector size changes
3. Register the feature in `FIXED_FEATURE_SET` or `SIMPLE_FEATURE_SET` as appropriate
4. Add a leakage test in `tests/data_integrity/`
5. Run `pytest tests/ -m "leakage" -x` to verify no temporal leakage
6. Run `pytest tests/ -m "unit" -x` for regression check

## Skill: Add a New Scraper

1. Create the scraper in `src/data/scrapers/`
2. Add ingestion integration in `src/data/ingestion/`
3. Register the CLI command in `src/main.py` if needed
4. Add rate limiting and error handling for external requests
5. Write tests with mocked HTTP responses — never hit live endpoints in tests

## Skill: Modify the ML Ensemble

1. Edit `src/ml/ensemble/` for model changes
2. If adding a new model type, integrate it into the ensemble voting/stacking logic
3. Update calibration in `src/ml/calibration/` if probability outputs change
4. Run `pytest tests/ -m "calibration" -x` to verify calibration still holds
5. **Never** change production-locked settings in `configs/production_2026.json`

## Skill: Run Production Pipeline

```bash
# Dry run (validation only, no predictions)
python src/run_production_2026.py --dry-run

# Full production run
python src/run_production_2026.py
```

**Do not** use `march-madness sota` for production — it is blocked from acting as production.
**Do not** modify locked fields in `configs/production_2026.json`.

## Skill: Data Ingestion

```bash
# Historical data (one-time or refresh)
march-madness ingest-historical --start-season 2005 --end-season 2025

# Roster data
march-madness scrape-rosters --start-year 2005 --end-year 2026
march-madness enrich-rosters --start-year 2005 --end-year 2026

# Current year
march-madness ingest --year 2026
```

## Skill: Generate Kaggle Submission

```bash
march-madness kaggle-export
march-madness backtest-kaggle  # Validate against historical results
```

Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Skill: Validate Reproducibility

```bash
march-madness freeze-pipeline    # Create reproducibility artifact
march-madness verify-freeze      # Verify against frozen artifact
march-madness audit-rdof         # Audit researcher degrees of freedom
```

## Skill: Add a New Test

1. Create `tests/test_<name>.py`
2. Markers are auto-assigned by `tests/conftest.py` based on file path — no need to manually decorate
3. Use fixtures from `conftest.py` (team data, predictions, outcomes, configs)
4. For data integrity tests, place in `tests/data_integrity/`
5. Run your new test: `pytest tests/test_<name>.py -v`
