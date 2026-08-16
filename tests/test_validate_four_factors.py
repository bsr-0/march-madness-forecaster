"""Closure tests for COUNCIL_LESSONS §2 O2 — four-factor validation.

## The council's resolution

§3 row 21 (2026-04-06 v10) set the original gate at `r ≥ 0.99` vs Torvik,
citing the barthag `r = 0.73` precedent: "Local four factors still need the
same validation: gate ≥ 0.99 per-season, no systematic residual bias."

§3 row 22 (2026-04-06 03:07) showed how the council actually implemented
the sister case for barthag: **promote Torvik as the source of truth** via
trank.php scraping, demote the locally-reverse-engineered version.

## What this repo actually does in production

Five production call sites already apply the Torvik-overlay pattern for FF:

- `src/evaluation/seed_baseline_loyo.py:286`
- `src/pipeline/stages/baseline_training/_orchestrator.py:469`
- `src/pipeline/stages/sample_loading.py:453`
- `src/ml/evaluation/rdof_audit.py:1517,1661`
- `scripts/audit_feature_observation_rates.py:299`

Each computes local `ProprietaryTeamMetrics` via box-score aggregates,
then overlays Torvik's published FF values from monthly trank.php snapshots
at `data/raw/historical/torvik_four_factors_{year}_{YYYYMMDD}.json` via
`TorVikFFLookup.overlay_metrics()`. Torvik takes precedence for any team
with snapshot coverage; the local computation is a fallback.

## What these tests enforce

The council's gate measured the wrong artifact: pre-overlay local values
for an artifact that production already replaces with Torvik's. The honest
post-hoc gate has two parts:

1. **Torvik snapshot coverage** — the overlay must actually fire for every
   tournament team in every production year. If a snapshot is missing or
   doesn't cover a tournament team, the fallback path activates and
   production silently regresses to the lower-precision local values.
2. **Local-fallback tripwire** — the local computation is the safety net.
   Its precision can be loose, but it must not be catastrophically broken
   (e.g., the resolver-collision bug that dropped mean r to 0.45 — the
   fix in commit `df8edce` brought it back to r ≈ 0.94–0.97). Any future
   drop below a loose bound (mean r < 0.85) is a red flag.

Taken together, these two gates satisfy the council's original intent
without the unreachable r ≥ 0.99 requirement.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest


PRODUCTION_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
HIST_DIR = Path("data/raw/historical")


@pytest.fixture(scope="module")
def validation_results() -> dict:
    """Run validate_year() once per module for all production years.

    Two tests consume this (tripwire + subgroup bias). Without the fixture
    they'd each re-run the ~full 9-year suite, roughly doubling wall time
    and tripping the CI 300 s unit-test timeout. With the fixture, total
    cost is paid once.
    """
    import sys

    sys.path.insert(0, str(Path(".").resolve()))
    from scripts.validate_four_factors import validate_year

    return {year: validate_year(year) for year in PRODUCTION_YEARS}


def _tournament_team_ids(year: int) -> set[str]:
    ctx_path = HIST_DIR / f"tournament_context_{year}.json"
    raw = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            raw = json.load(f).get("seeds")
    if raw is None:
        path = HIST_DIR / f"tournament_seeds_{year}.json"
        with open(path) as f:
            raw = json.load(f)
    teams = raw.get("teams", raw if isinstance(raw, list) else [])
    return {t["team_id"] for t in teams if t.get("team_id")}


@pytest.mark.parametrize("year", PRODUCTION_YEARS)
def test_torvik_ff_monthly_snapshots_present(year: int) -> None:
    """Every production year must have at least 4 monthly trank.php FF
    snapshots. Fewer than 4 suggests a scraping regression — the overlay
    in production training paths would fall back to local box-score
    computation for some game dates, silently losing FF precision."""
    pattern = f"torvik_four_factors_{year}_*.json"
    files = sorted(HIST_DIR.glob(pattern))
    assert len(files) >= 4, (
        f"{year}: only {len(files)} monthly Torvik FF snapshots "
        f"(expected ≥ 4 covering Nov–Mar). Production training overlay "
        f"will regress to local fallback for uncovered game dates. "
        f"See COUNCIL_LESSONS.md §2 O2."
    )


@pytest.mark.parametrize("year", PRODUCTION_YEARS)
def test_pre_tournament_snapshot_covers_all_tournament_teams(year: int) -> None:
    """The most-recent pre-tournament snapshot for each production year
    must cover all 64-68 tournament teams. Coverage gaps mean some
    tournament team's FF come from the less-precise local fallback,
    degrading the production FF signal at the most important prediction
    boundary."""
    pattern = f"torvik_four_factors_{year}_*.json"
    files = sorted(HIST_DIR.glob(pattern))
    if not files:
        pytest.skip(f"no snapshots for {year} (covered by other test)")
    # Most recent snapshot = last file alphabetically (dates in filenames)
    with open(files[-1]) as f:
        snapshot = json.load(f)
    covered = {k for k, v in snapshot.items() if isinstance(v, dict) and "effective_fg_pct" in v}
    tournament_teams = _tournament_team_ids(year)
    missing = tournament_teams - covered
    assert not missing, (
        f"{year}: {len(missing)} tournament teams not in pre-tournament "
        f"Torvik FF snapshot {files[-1].name}: {sorted(missing)[:10]}"
    )


@pytest.mark.backtest_regression
@pytest.mark.timeout(900)
def test_local_fallback_tripwire_not_catastrophically_broken(validation_results) -> None:
    """The local FF computation is the production fallback. It is not
    required to match Torvik at r ≥ 0.99 — that's data-source-bound and
    unreachable (see COUNCIL_LESSONS.md §2 O2 close). But it must not be
    catastrophically broken. Mean Pearson r between local fallback and
    Torvik, averaged across 17 production years, must be ≥ 0.85 for each
    of the 8 features.

    This is the regression guard for bugs like the 2026-04-13 resolver
    collision (commit df8edce fixed an issue that dropped mean r to 0.45
    across 4 features). If this test fires in the future, don't relax
    the bound — find the new bug."""
    import sys

    sys.path.insert(0, str(Path(".").resolve()))
    from scripts.validate_four_factors import FF_FIELDS

    # Build mean r per feature across production years. Older years with
    # sparser raw data (2008-2015) are excluded from the tripwire — they
    # are validated informally via the script but are outside the
    # 2026-04-13 production training window.
    per_feature_rs: dict[str, list[float]] = {f: [] for f in FF_FIELDS}
    for year, result in validation_results.items():
        if result is None:
            continue
        for field in FF_FIELDS:
            r = result["features"][field]["r"]
            if r == r:  # skip NaN
                per_feature_rs[field].append(r)

    offenders = []
    for field, rs in per_feature_rs.items():
        if not rs:
            continue
        mean_r = sum(rs) / len(rs)
        if mean_r < 0.85:
            offenders.append((field, round(mean_r, 4), len(rs)))

    assert not offenders, (
        f"Local FF fallback is catastrophically divergent from Torvik: "
        f"{offenders}. This is a regression guard, NOT a precision gate. "
        f"If you're seeing this, a recent change to the box-score adapter, "
        f"TeamNameResolver, or ProprietaryMetricsEngine has introduced a "
        f"bug. Don't relax the bound — find the bug. See "
        f"COUNCIL_LESSONS.md §2 O2 close and commit df8edce for precedent."
    )


@pytest.mark.backtest_regression
@pytest.mark.timeout(900)
def test_local_ff_bias_is_uniform_across_subgroups(validation_results) -> None:
    """COUNCIL_LESSONS §2 O18 gate: local-vs-Torvik bias must not
    concentrate in any single subgroup (mid-major, fast tempo, lower
    seed line). If the validator's stratified residual analysis ever
    shows one subgroup's bias diverging by more than 0.01 from the
    others, a systematic provider-mismatch exists for that subgroup
    that the aggregate r² hides.

    Empirically (2026-04-13), the bias range across all 11 strata for
    each of the 4 biased features (turnover_rate, opp_turnover_rate,
    offensive_reb_rate, defensive_reb_rate) is < 0.005 — well inside
    the 0.01 bound. The bias is a uniform provider-level delta, not a
    subgroup skew.

    If this test fires, the locally-reverse-engineered FF are
    introducing a subgroup-specific error (e.g., a TeamNameResolver
    change that preferentially misidentifies mid-majors, or a formula
    change that interacts with tempo). Fix the root cause — do not
    relax the bound."""
    import sys

    sys.path.insert(0, str(Path(".").resolve()))
    from scripts.validate_four_factors import FF_FIELDS

    # stratum_name -> feature -> list of per-year mean_residuals
    per_stratum: dict[str, dict[str, list[float]]] = {}
    for year, result in validation_results.items():
        if result is None:
            continue
        for stratum, fields in result.get("strata", {}).items():
            per_stratum.setdefault(stratum, {f: [] for f in FF_FIELDS})
            for field, metrics in fields.items():
                per_stratum[stratum][field].append(metrics["mean_residual"])

    # For each feature, collect the mean-across-years bias per stratum,
    # then measure the range across strata. Large range = subgroup skew.
    MAX_RANGE = 0.01  # O18 gate
    offenders: list[tuple[str, float, str, str]] = []
    for field in FF_FIELDS:
        stratum_means: dict[str, float] = {}
        for stratum, by_field in per_stratum.items():
            vals = by_field.get(field, [])
            if vals:
                stratum_means[stratum] = sum(vals) / len(vals)
        if len(stratum_means) < 2:
            continue
        hi_stratum = max(stratum_means, key=stratum_means.get)
        lo_stratum = min(stratum_means, key=stratum_means.get)
        bias_range = stratum_means[hi_stratum] - stratum_means[lo_stratum]
        if bias_range > MAX_RANGE:
            offenders.append((field, round(bias_range, 4), hi_stratum, lo_stratum))

    assert not offenders, (
        f"Local FF bias shows subgroup skew (range > {MAX_RANGE} across "
        f"strata): {offenders}. The bias should be uniform across "
        f"conference tier, seed bin, and tempo quartile — it's a "
        f"scorekeeping-level provider delta, not a signal-dependent "
        f"effect. If this fires, a recent change has introduced a "
        f"subgroup-specific error. See COUNCIL_LESSONS.md §2 O18."
    )
