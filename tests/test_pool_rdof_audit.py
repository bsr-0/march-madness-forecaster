"""Gates on the pool-search RDoF audit (audit finding H2 / recommendation 12).

The deleted `src/ml/evaluation/rdof_audit.py` had NO tests at all, and the
consequence is visible in its own source: a hand-maintained constant registry
that drifted from the code it claimed to describe until it needed a section
titled "Previously Unregistered Constants", plus a hand-copied
`_N_TUNED_CONSTANTS = 58` in a second module. A registry that can be silently
wrong about the code is worse than no registry, because it launders a stale
claim as a checked one.

So the two load-bearing tests here are not about statistics:

  * `test_registry_matches_live_code` resolves every registered symbol by
    import and compares. The registry cannot describe a value the code does
    not have.
  * `test_every_module_level_constant_is_classified` fails when a new constant
    appears in the searched modules and is neither registered nor explicitly
    declared not-a-knob. Drift by ADDITION is what actually happened last time.

The rest gate the statistics (a family of pure nulls must not produce a
significant winner), the holdout enforcement, and the report's ability to
render from a partial measurement set.
"""

from __future__ import annotations

import json
import re

import pytest

from src.governance.pool_rdof_audit import (
    ALL_STATUSES,
    PARAMETER_CLEAN_HOLDOUT,
    REGISTRY,
    SEQUESTERED_YEARS,
    STATUS_REMOVED_AFTER_MEASURING,
    HoldoutContaminationError,
    PoolRDOFReport,
    UnresolvableSymbol,
    _NOT_A_DOF,
    assert_not_sequestered,
    check_holdout_contamination,
    dof_summary,
    drift_report,
    live_value,
    record_holdout_evaluation,
    registry_hash,
)

# Modules whose module-level constants must all be classified. Kept short on
# purpose: these are the two files that define the searched strategy.
_SEARCHED_MODULES = (
    "scripts/mc_pool_backtest.py",
    "src/optimization/poolaware_recipe.py",
)

# A module-level constant assignment: SCREAMING_CASE (optionally leading
# underscore), optionally annotated, at column 0.
_CONSTANT_RE = re.compile(r"^(_?[A-Z][A-Z0-9_]{2,})\s*(?::[^=\n]+)?=", re.M)


# ---------------------------------------------------------------------------
# The registry describes the code
# ---------------------------------------------------------------------------


def test_registry_matches_live_code():
    """Every registered value that names a live symbol must equal it.

    This is the whole reason the registry stores `live_symbol` instead of
    transcribing values: transcription is what drifted last time.
    """
    report = drift_report()
    assert report["drifted"] == [], (
        "The RDoF registry no longer describes the code. Either the code changed (update "
        "src/governance/pool_rdof_audit.py REGISTRY, and note that changing a registered "
        "knob invalidates any recorded holdout evaluation -- see check_holdout_contamination) "
        f"or the registry was wrong. Drifted: {report['drifted']}"
    )


def test_drift_check_actually_covers_most_of_the_registry():
    """Guard against the registry passing by checking nothing.

    `test_registry_matches_live_code` is vacuously true if every entry is an
    unresolvable inline literal. Coverage is currently ~70%; the floor is set
    below that so ordinary additions don't trip it, but a collapse does.
    """
    report = drift_report()
    assert report["n_checked"] >= 25, f"only {report['n_checked']} entries machine-checked; the gate has gone hollow"
    assert report["coverage"] >= 0.60, f"drift coverage fell to {report['coverage']:.0%}"


def test_every_module_level_constant_is_classified():
    """A new constant in the searched modules must be registered or excused.

    Drift by ADDITION is the failure the deleted module actually suffered --
    not a wrong value, but a knob nobody had listed. Registering it or naming
    it in `_NOT_A_DOF` are both fine; silence is not.
    """
    registered_paths = " | ".join(d.code_path for d in REGISTRY)
    unclassified = []
    for path in _SEARCHED_MODULES:
        with open(path) as f:
            source = f.read()
        for name in sorted(set(_CONSTANT_RE.findall(source))):
            if name in _NOT_A_DOF:
                continue
            if re.search(rf"\b{re.escape(name)}\b", registered_paths):
                continue
            unclassified.append(f"{path}:{name}")
    assert not unclassified, (
        "Module-level constant(s) in the pool-strategy search are neither in the RDoF "
        "registry nor in the _NOT_A_DOF allowlist: "
        f"{unclassified}. Add a PoolDegreeOfFreedom entry (with its provenance) or, if it is "
        "genuinely not a knob, an _NOT_A_DOF entry saying why."
    )


def test_registry_entries_are_well_formed():
    """Structural invariants, so a malformed entry fails here not in the report."""
    names = [d.name for d in REGISTRY]
    assert len(names) == len(set(names)), "duplicate registry names"
    for d in REGISTRY:
        assert d.tier in (1, 2, 3), f"{d.name}: bad tier {d.tier}"
        assert d.status in ALL_STATUSES, f"{d.name}: bad status {d.status}"
        assert d.derivation.strip(), f"{d.name}: empty derivation — provenance is the point of the registry"
        assert d.code_path.strip(), f"{d.name}: empty code_path"
        if d.live_symbol:
            assert ":" in d.live_symbol, f"{d.name}: live_symbol must be 'module:attr'"


def test_forking_paths_debt_is_recorded():
    """The removed candidate families must stay in the registry.

    They are the part of finding H2 that cannot be corrected statistically --
    the code is gone, so no resampling scheme can put them back in the family.
    If they vanish from the registry the audit silently understates itself.
    """
    removed = [d.name for d in REGISTRY if d.status == STATUS_REMOVED_AFTER_MEASURING]
    assert len(removed) >= 2, f"expected the recorded removed-after-measuring families, found {removed}"
    for d in REGISTRY:
        if d.status == STATUS_REMOVED_AFTER_MEASURING:
            assert 2026 in d.contaminated_by, (
                f"{d.name}: a family removed on the strength of a 15-season aggregate saw 2026's "
                "outcome; contaminated_by must record it"
            )


def test_live_value_distinguishes_cannot_check_from_wrong():
    """An inline literal must raise, not silently compare equal to something."""
    inline = next(d for d in REGISTRY if d.live_symbol is None)
    with pytest.raises(UnresolvableSymbol):
        live_value(inline)


def test_duplicated_risk_grids_still_agree():
    """`recency_hparam_fitter` hand-copies the recipe's grids instead of importing.

    Registered as tier 2 on the grounds that it is not an independent choice.
    That is only true while the copies match, so make the copy load-bearing
    rather than latent.
    """
    from src.optimization import poolaware_recipe, recency_hparam_fitter

    assert tuple(recency_hparam_fitter._REGION_RISK_LEVELS) == tuple(poolaware_recipe.POOLAWARE_RISK_LEVELS), (
        "recency_hparam_fitter._REGION_RISK_LEVELS has drifted from "
        "poolaware_recipe.POOLAWARE_RISK_LEVELS. They are hand-copies, and the fitter tunes "
        "against whichever it holds — import the recipe's constant instead of re-declaring it."
    )
    assert tuple(recency_hparam_fitter._EXHAUSTIVE_RISK_LEVELS) == tuple(
        poolaware_recipe.POOLAWARE_EXHAUSTIVE_RISKS
    ), "recency_hparam_fitter._EXHAUSTIVE_RISK_LEVELS has drifted from POOLAWARE_EXHAUSTIVE_RISKS"


# ---------------------------------------------------------------------------
# Holdout sequestration
# ---------------------------------------------------------------------------


def test_sequestered_year_is_absent_from_every_search_window():
    """2027 must not be reachable by the default backtest or aggregate paths."""
    from scripts.mc_pool_backtest import BACKTEST_YEARS, EVALUATION_YEARS

    for year in SEQUESTERED_YEARS:
        assert year not in BACKTEST_YEARS, f"{year} is sequestered but appears in BACKTEST_YEARS"
        assert year not in EVALUATION_YEARS, f"{year} is sequestered but appears in EVALUATION_YEARS"


def test_assert_not_sequestered_raises_for_the_holdout_and_passes_otherwise(tmp_path, monkeypatch):
    """Enforcement, not assertion: passing 2027 to a sweep must fail loudly."""
    import src.governance.pool_rdof_audit as mod

    monkeypatch.setattr(mod, "HOLDOUT_LOCKFILE", tmp_path / "lock.json")

    assert_not_sequestered([2011, 2025], context="unit test")  # must not raise

    with pytest.raises(HoldoutContaminationError, match="sequestered"):
        assert_not_sequestered([2024, 2027], context="unit test")


def test_holdout_lockfile_detects_a_knob_change_after_evaluation(tmp_path, monkeypatch):
    """The lockfile's entire purpose, exercised end to end.

    Once a holdout has been scored, changing a registered knob means the
    configuration that produced that result no longer exists — so the result
    has stopped describing the shipped system. That must be detectable, or
    "we evaluated the holdout" degrades into a claim about the past.
    """
    import src.governance.pool_rdof_audit as mod

    monkeypatch.setattr(mod, "HOLDOUT_LOCKFILE", tmp_path / "lock.json")

    entry = record_holdout_evaluation(year=2026, evidence_level="2.5", summary={"p_first": {"x": 0.1}})
    assert entry["registry_hash"] == registry_hash()
    assert check_holdout_contamination() == [], "a freshly recorded evaluation cannot be contaminated"

    # Simulate a knob change by swapping in a registry with one different value.
    mutated = tuple(
        mod.PoolDegreeOfFreedom(**{**d.__dict__, "current_value": 999}) if d.name == "n_opponents" else d
        for d in mod.REGISTRY
    )
    monkeypatch.setattr(mod, "REGISTRY", mutated)

    stale = check_holdout_contamination()
    assert len(stale) == 1, "changing a registered knob after a holdout evaluation must be flagged"
    assert stale[0]["year"] == 2026
    assert "no longer describes the shipped system" in stale[0]["message"]


def test_parameter_clean_holdout_is_labelled_below_level_one():
    """2026 must never be reported as an untouched holdout.

    It cannot be one: the candidate-family set was pruned on aggregates that
    included it (2026-05-03 and 2026-05-16), months before
    CONTAMINATED_EVAL_YEARS existed (2026-09-09). The report has to say so.
    """
    from src.governance.pool_rdof_audit import holdout_status

    status = holdout_status()
    assert PARAMETER_CLEAN_HOLDOUT not in SEQUESTERED_YEARS
    why = status["level_2_5_parameter_clean"]["why_not_level_1"]
    assert "2026-05-03" in why and "a3fb412" in why, "the contamination evidence must stay in the report, not just in a commit message"
    assert status["no_historical_holdout"]


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------


def test_stepdown_detects_a_genuine_winner_among_nulls():
    np = pytest.importorskip("numpy")
    from src.evaluation.multiplicity import romano_wolf_stepdown

    rng = np.random.default_rng(0)
    diffs = np.vstack([rng.normal(0.08, 0.03, size=(1, 14)), rng.normal(0.0, 0.03, size=(30, 14))])
    names = ["winner"] + [f"null{i}" for i in range(30)]

    result = romano_wolf_stepdown(diffs, names, n_resamples=2000)

    assert result.best == "winner"
    assert result.p_adjusted["winner"] < 0.05, "a 2.7-sigma effect must survive correction over 31 arms"


def test_stepdown_does_not_crown_a_winner_in_an_all_null_family():
    """The correction's reason for existing.

    With 31 pure-null arms the best one will have a small UNCORRECTED p by
    construction. If the corrected p were also small, the whole measurement
    would be decoration.
    """
    np = pytest.importorskip("numpy")
    from src.evaluation.multiplicity import romano_wolf_stepdown

    rng = np.random.default_rng(7)
    diffs = rng.normal(0.0, 0.03, size=(31, 14))
    names = [f"null{i}" for i in range(31)]

    result = romano_wolf_stepdown(diffs, names, n_resamples=2000)

    best_unadjusted = min(result.p_unadjusted.values())
    assert best_unadjusted < 0.10, "sanity check on the fixture: the best of 31 nulls should look good raw"
    assert result.p_max_statistic > 0.20, (
        f"best-of-family p was {result.p_max_statistic} for a family of pure nulls; "
        "the multiplicity correction is not correcting"
    )


def test_stepdown_controls_family_wise_error_under_the_global_null():
    """Empirical FWER at alpha=0.10 over repeated all-null families."""
    np = pytest.importorskip("numpy")
    from src.evaluation.multiplicity import romano_wolf_stepdown

    rejections = 0
    trials = 40
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        diffs = rng.normal(0.0, 0.03, size=(12, 14))
        result = romano_wolf_stepdown(diffs, [f"m{i}" for i in range(12)], n_resamples=500, seed=s)
        if min(result.p_adjusted.values()) < 0.10:
            rejections += 1
    # 0.10 nominal; allow generous slack for 40 trials (binomial SE ~4.7pp).
    assert rejections / trials <= 0.25, f"empirical FWER {rejections}/{trials} far exceeds the nominal 0.10"


def test_stepdown_adjusted_p_is_monotone_in_the_ordering():
    np = pytest.importorskip("numpy")
    from src.evaluation.multiplicity import romano_wolf_stepdown

    rng = np.random.default_rng(3)
    diffs = rng.normal(0.02, 0.03, size=(15, 14))
    names = [f"m{i}" for i in range(15)]

    result = romano_wolf_stepdown(diffs, names, n_resamples=1000)

    ordered = sorted(names, key=lambda n: -result.t_stat[n])
    adj = [result.p_adjusted[n] for n in ordered]
    assert adj == sorted(adj), "stepdown-adjusted p-values must never decrease down the ordering"


def test_paired_matrix_refuses_a_run_missing_the_headline_mode():
    """A partially-failed run must not be silently corrected as a smaller family."""
    from src.evaluation.multiplicity import paired_matrix_from_results

    records = [
        {"year": 2011, "mode": "seed", "p_first": 0.04},
        {"year": 2012, "mode": "seed", "p_first": 0.04},
        {"year": 2011, "mode": "meta_region_poolaware", "p_first": 0.12},
        # 2012 missing for the headline mode
    ]
    with pytest.raises(ValueError, match="do not cover all"):
        paired_matrix_from_results(records, "seed", require_modes=["meta_region_poolaware"])


def test_circular_modes_are_flagged_not_silently_dropped():
    """A mode graded by its own probability source must be labelled, and kept.

    `fixed_blendA100_r35` is alpha=1.0, i.e. pure seed probabilities, and the
    referee that draws every simulated tournament is the same seed model. It
    tops the aggregate P(1st) table for that reason (finding H11), so quoting
    it as a rival to the headline would be wrong. Excluding it from the family
    would ALSO be wrong — choosing the comparison set after seeing the results
    is the exact H2 behaviour this audit exists to measure. So: flagged, kept.
    """
    from scripts.pool_rdof_audit import CIRCULAR_MODES

    assert "fixed_blendA100_r35" in CIRCULAR_MODES
    assert "seed" in CIRCULAR_MODES["fixed_blendA100_r35"], "the flag must say WHY it is circular"

    from scripts.mc_pool_backtest import ALL_MODES

    for mode in CIRCULAR_MODES:
        assert mode in ALL_MODES, (
            f"{mode} is flagged circular but is no longer in ALL_MODES; the flag has gone stale"
        )


def test_stepdown_rejects_a_single_season():
    from src.evaluation.multiplicity import romano_wolf_stepdown

    with pytest.raises(ValueError, match="at least 2 seasons"):
        romano_wolf_stepdown([[0.1]], ["m"], n_resamples=100)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_renders_and_serialises_with_no_measurements():
    """A registry-only run must produce a complete, honest report.

    Partial measurement sets are the normal case while the long runs are
    executing, and a report that crashes on an absent section is a report
    nobody runs.
    """
    report = PoolRDOFReport()

    text = report.to_text()
    assert "RESEARCHER DEGREES OF FREEDOM" in text
    assert "not measured" in text, "an unmeasured section must say so rather than render blank"

    payload = report.to_dict()
    json.dumps(payload)  # must be serialisable
    assert payload["metadata"]["audit_recommendation"] == 12
    assert len(payload["registry"]) == len(REGISTRY)
    assert any(r.startswith("DoF RATIO:") for r in payload["recommendations"])
    assert any(r.startswith("CIRCULARITY:") for r in payload["recommendations"])
    assert any(r.startswith("HOLDOUT:") for r in payload["recommendations"])


def test_to_dict_has_no_side_effects():
    """Serialising twice must give the same thing.

    The deleted module's `to_dict()` called `_generate_recommendations()` as a
    side effect of serialisation, which made the artifact depend on how many
    times it had been written.
    """
    report = PoolRDOFReport()
    first = json.dumps(report.to_dict()["recommendations"])
    second = json.dumps(report.to_dict()["recommendations"])
    assert first == second


def test_dof_summary_discounts_measured_flat_knobs_but_not_unmeasured_ones():
    """A flat knob cost ~no degrees of freedom; an unmeasured one is not free."""
    baseline = dof_summary()
    assert baseline["effective_dof_spent"] == baseline["raw_dof_spent"], "unmeasured knobs must count at full weight"
    assert baseline["effective_dof_is_measured"] is False

    flattened = dof_summary({"pa_trials": 0.0})
    assert flattened["effective_dof_spent"] < baseline["effective_dof_spent"]
    assert flattened["effective_dof_is_measured"] is True


def test_dof_ratio_uses_seasons_not_repeat_trials():
    """Audit H1's finding, encoded.

    A CI or ratio over bracket-repeats understates uncertainty roughly
    threefold, because repeats within a season share one seed layout and one
    pick distribution. The denominator here must be 14, not 1400.
    """
    from scripts.mc_pool_backtest import EVALUATION_YEARS

    summary = dof_summary()
    assert summary["n_evaluation_seasons"] == len(EVALUATION_YEARS) == 14
