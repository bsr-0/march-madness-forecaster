# March Madness Forecaster

Pool-aware NCAA tournament bracket forecasting and research. The project models game outcomes and pool-opponent pick behavior, then builds and evaluates bracket strategies under specified pool, scoring, and payout settings. A strong candidate score is not, by itself, evidence of out-of-sample superiority or a production recommendation.

## What this project does

- ingests historical game and rating data
- builds game-level win-probability models
- represents pool-opponent pick behavior and payout structure
- builds and scores bracket strategies for objectives such as expected points and probability of finishing first
- keeps research/CLI optimization distinct from the production site's frozen recommendation and release gates

## Repository layout

- `src/` — production code for data, simulation, optimization, and evaluation
- `scripts/` — reusable scripts for ingestion, backtests, and reporting
- `docs/` — active documentation and archived report history
- `artifacts/` — generated results, audit outputs, and historical reports
- `tests/` — validation and regression coverage

## Quick start

```bash
pip install -e .

python -m src.main --help
python -m src.main ingest --year 2027
python -m src.main optimize-pool --year 2026 --pool-size 30
```

`ingest` collects source data; it does not create or approve a 2027 release. `optimize-pool` is a separate CLI analysis and does not generate the production-site payload. Follow the [2027 roadmap](docs/PROJECT_ROADMAP_2027.md) and [operations guide](docs/OPERATIONS.md) for release selection, fallback, and build procedures. The commands above were checked against the current CLI help; ingestion and optimization themselves were not run as part of this documentation review.

## Canonical documentation

- [docs/PROJECT_ROADMAP_2027.md](docs/PROJECT_ROADMAP_2027.md) — release phases, gates, and scope control
- [docs/METHODOLOGY_AND_REPORTS.md](docs/METHODOLOGY_AND_REPORTS.md) — methodology, validation, and report governance
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — analytics, pool settings, storage, delivery, and site-review notes
- [docs/archive/README.md](docs/archive/README.md) — preserved historical documents and superseded notes

## Working rules

- Keep the root directory lean and stable.
- Treat historical reports in the archive as context, not current truth.
- Prefer small, targeted changes and add tests for behavior changes.
- When a bug is found, stop, verify the cause, and fix the underlying defect rather than masking it.

## Current repo posture

This repo contains both active operational docs and historical research artifacts. The active project story lives in the root-level docs and in `docs/`; archived reports remain preserved for context but should not be treated as current workflow guidance.
