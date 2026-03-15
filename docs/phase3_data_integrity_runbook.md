# Phase 3 Data Integrity Runbook

Step-by-step guide for running, maintaining, and extending the Phase 3 data integrity checks.

---

## 1. Running All Checks Locally

```bash
# Full data integrity suite
pytest tests/data_integrity/ -v --tb=short

# Individual test files
pytest tests/data_integrity/test_point_in_time_contracts.py -v
pytest tests/data_integrity/test_training_row_assembly.py -v
pytest tests/data_integrity/test_leakage_rules.py -v
pytest tests/data_integrity/test_lineage_manifests.py -v
pytest tests/data_integrity/test_provider_robustness.py -v

# Contract validator standalone
python -c "
from src.data.contract_validator import ContractValidator
v = ContractValidator()
r = v.validate()
print(r.summary())
"
```

---

## 2. Adding a New Feature

When adding a new feature to `TeamFeatures`, you must:

### Step 1: Add contract entry

Edit `contracts/feature_contracts.yaml` and add an entry with ALL required fields:

```yaml
- feature_name: my_new_feature
  owner: data-platform
  source:
    system: proprietary_metrics  # or torvik, roster, game_flows, etc.
    dataset: game_records
    field: my_new_field
  event_timestamp_field: game_end_time
  available_at_logic:
    type: offset_from_event      # or function_ref, static
    event_timestamp_field: game_end_time
    offset_hours: 0.5
  transformation_logic_ref: "src/data/features/feature_engineering.py::FeatureEngineer.extract_team_features"
  prediction_time_reference: scheduled_tipoff_time_utc
  allowed_lag: ">=0h"
  null_handling_policy: "use_default_0.0"
  leakage_checks:
    same_game_outcome_leakage: false
    future_opponent_outcome_leakage: false
    post_game_aggregate_leakage: false
    tournament_info_leakage_pre_tournament: false
  contract_version: "1.0.0"
  test_ids: ["PIT-NNN", "LEAK-R1-NNN"]
  risk_tier: medium   # low, medium, high, or critical
  snapshot_policy: point_in_time_snapshot_required
  revision_behavior: latest_forbidden
  late_arrival_policy: exclude_if_available_after_prediction_time
```

### Step 2: Add leakage test

Add at least one test per applicable leakage rule in `tests/data_integrity/test_leakage_rules.py`.

Evidence requirements by risk tier:
- **low**: structural evidence sufficient
- **medium**: structural + synthetic required
- **high**: structural + synthetic + historical required
- **critical**: all three + CODEOWNERS approval

### Step 3: Update manifest

If the feature adds a new data source, update `src/data/manifest_generator.py` to include it in `raw_inputs`.

### Step 4: Validate

```bash
pytest tests/data_integrity/test_point_in_time_contracts.py -v
```

This will catch:
- Missing contract (forward completeness)
- Invalid schema fields
- Missing test_ids
- Broken transformation_logic_ref

---

## 3. Interpreting Failures

### Contract validation failures

| Error | Meaning | Fix |
|-------|---------|-----|
| `Forward completeness: Active feature has no contract` | A feature in `get_feature_names()` is missing from contracts YAML | Add contract entry |
| `Reverse completeness: Contract exists but feature not active` | Orphan contract for removed feature | Mark as `deprecated: true` with `sunset_date` |
| `Matchup lineage: Derived feature references base without contract` | A `diff_*` or `abs_*` feature has no underlying team-level contract | Add contract for the base feature |
| `available_at_executable: Invalid type` | available_at_logic uses unsupported type | Use `offset_from_event`, `function_ref`, or `static` |
| `transformation_ref_exists: File not found` | Referenced source file was moved/deleted | Update the ref path |

### Leakage test failures

| Error pattern | Meaning | Fix |
|---------------|---------|-----|
| `LEAKAGE: features at game_date include game outcome` | Same-game outcome in features (Rule 1) | Verify `compute_as_of` uses strict `<` |
| `LEAKAGE: opponent future game included` | Future opponent data (Rule 2) | Verify single `compute_as_of` call per row |
| `LEAKAGE: adding game at cutoff date changed features` | Post-date aggregate (Rule 3) | Verify expanding means exclude current |
| `Seeds must be zeroed before tournament cutoff` | Tournament info pre-tournament (Rule 4) | Gate seed attachment on `tournament_cutoff` |

### Manifest failures

| Level | Error | Fix |
|-------|-------|-----|
| A | `Missing or empty field: X` | Populate the field in manifest generator |
| B | `training_data hash mismatch` | Inputs changed between runs — verify data pipeline determinism |
| C | `prediction output mismatch` | Non-deterministic backend — pin seeds or accept as aspirational |

### Robustness failures

| Error | Meaning | Fix |
|-------|---------|-----|
| `flip_rate exceeds threshold` | Model too dependent on a single provider | Add fallback providers or regularize |
| `calibration_drift exceeds threshold` | Provider removal shifts probability estimates | Investigate feature importance and add robustness |
| `performance_drop exceeds threshold` | Brier score degradation too large | Review feature group dependency |

---

## 4. Updating Robustness Baselines

After a successful CI run with robustness tests:

```python
from tests.data_integrity.test_provider_robustness import save_baseline

# Collect observed metrics from test runs
metrics = {
    "flip_rate_increase": [0.03, 0.04, 0.035],
    "calibration_drift": [0.01, 0.012, 0.011],
    "performance_drop": [0.02, 0.025, 0.022],
}
save_baseline(metrics)
```

This updates `artifacts/robustness_baseline.json` and future runs will use adaptive thresholds (mean + 2σ).

---

## 5. CODEOWNERS Workflow for Critical Features

Critical features (`seed_strength`, `external_rating_composite`, `external_rating_spread`, `conf_tourney_champ`) are gated by `.github/CODEOWNERS`.

When modifying these features:
1. PR must be reviewed and approved by `@data-integrity-reviewers`
2. All HIST-* tests must pass
3. Contract changes must include updated `contract_version`
4. Reviewer must verify `snapshot_policy`, `revision_behavior`, and `late_arrival_policy` are appropriate

---

## 6. Evidence Type Reference

| Marker | Meaning | Strength |
|--------|---------|----------|
| `@pytest.mark.evidence_structural` | Source code inspection | Catches code patterns |
| `@pytest.mark.evidence_synthetic` | Injected canary detection | Catches logic bugs |
| `@pytest.mark.evidence_historical` | Real historical data verification | Strongest proof |

Minimum requirements by risk tier:
- **low**: structural
- **medium**: structural + synthetic
- **high**: structural + synthetic + historical
- **critical**: all three + CODEOWNERS review
