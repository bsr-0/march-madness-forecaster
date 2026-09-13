"""RDoF audit over the pool-strategy search (audit recommendation 12 / finding H2).

WHAT THIS ANSWERS
-----------------
The headline is that ``meta_region_poolaware`` finishes first in ~12% of
simulated pools. That mode was chosen as the best of ``ALL_MODES`` measured on
the same 14 seasons, using a search whose knobs mostly have no recorded
provenance. Audit finding H2 ("Garden of forking paths with no untouched
holdout") says nothing in the repo measures what that costs. Three questions,
three subcommands:

  ``--registry-only``  How many degrees of freedom did the search spend, what
                       is each one's provenance, and does the registry still
                       match the code? (No simulation; seconds.)

  ``--multiplicity``   Is the headline just the best of many? Runs every mode
                       in ``ALL_MODES`` over the 14 evaluation seasons and
                       applies a Romano-Wolf stepdown to the P(1st) deltas
                       against the ``seed`` baseline.

  ``--sweep``          Does the headline depend on the knobs nobody justified?
                       A specification curve over the four tier-3 knobs with
                       real provenance debt, computed on 2011-2025 ONLY.

  ``--holdout-2026``   Evaluates the parameter-clean holdout. See
                       ``src.governance.pool_rdof_audit`` for why 2026 is
                       Level 2.5 and not Level 1, and why it cannot be made
                       Level 1.

PRE-REGISTERED DECISION RULES. Fixed here, in writing, before either long run
was executed -- the same discipline ``scripts/independent_referee_check.py``
follows, and for the same reason: a rule chosen after seeing the output is not
a rule.

  Multiplicity (on ``p_max_statistic``, the best-of-family adjusted p):
    < 0.05         the headline survives correction for selection over the family
    0.05 - 0.20    not distinguishable from selection noise at n=14
    > 0.20         the headline is substantially a selection artifact

  Sensitivity (per knob, on the spread of aggregate P(1st) across its grid):
    < 1.5pp        flat -- the knob did not matter, ~0 effective DoF spent
    1.5 - 3.0pp    mild slope, ~0.5 effective DoF
    > 3.0pp        sharp -- the knob is load-bearing and its lack of
                   provenance is a live problem, ~1.0 effective DoF
  (1.5pp is one season-level standard error on the headline, per audit H1.
  Judging "flat" against anything finer than the noise floor would be
  measuring the noise.)

DO NOT PASS ``--save-brackets`` ANYWHERE IN THIS FILE, and do not add it. That
flag rewrites the whole of ``artifacts/backtest_brackets/`` with only the modes
named on the run, and ``tests/test_oracle_drift_guard.py`` pins 39 tests to the
``torvik`` / ``f4_first_tv`` / ``e8_first_tv`` entries in those files. Doing this
by accident cost a 39-test regression while building audit recommendation 11.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.canonical_contract import CANONICAL_N_OPPONENTS

OUT_DIR = Path("artifacts/headline_measurement")

# The headline mode and its paired baseline. `seed` is the baseline every
# statistical test in mc_pool_backtest already uses, so keeping it here means
# the corrected number is comparable to the uncorrected one.
HEADLINE_MODE = "meta_region_poolaware"
BASELINE_MODE = "seed"

# Modes whose probability base IS the referee that grades every trial.
#
# `blend = alpha * seed_rp + (1 - alpha) * noseed_rp`, so alpha=1.0 is pure
# seed probabilities -- and the simulated-outcome referee is `seed_pw`, the
# same seed model. A bracket built from the grader's own beliefs scores well
# close to by definition, so its P(1st) measures audit finding C2's
# circularity rather than bracket construction. It is NOT excluded from the
# family (that would be choosing the comparison set after seeing the results,
# which is the H2 behaviour this whole audit is about) -- it is flagged, so it
# is never quoted as a rival to the headline. See finding H11.
CIRCULAR_MODES = {
    "fixed_blendA100_r35": (
        "alpha=1.0 is pure seed_rp, and the referee is seed_pw -- graded by its own source. "
        "Its alpha sweep shows the signature: .0879 / .1050 / .1086 / .1036 then a 1.8pp jump "
        "to .1214 at exactly the endpoint that coincides with the referee."
    ),
}

# The canonical measurement contract, from README "What the backtest number
# means". Every run in this file uses it, so a number from here is directly
# comparable to the published headline. Changing any of these makes the output
# incomparable, which is the whole failure mode audit finding H6 was about
# (n_opponents was 999 until 2026-09-10 and silently measured a 1000-person
# pool for every pre-2023 season).
CANONICAL = dict(
    n_opponents=CANONICAL_N_OPPONENTS,
    n_repeats=100,
    opponent_source="pool",
    team_identity=True,
    pa_trials=500,
)


def _canonical_run(
    modes: Sequence[str],
    years: Sequence[int],
    workers: int,
    **overrides: Any,
) -> List[Dict[str, Any]]:
    """One ``run_backtest`` at the canonical contract, returning raw records.

    Passing ``years`` explicitly makes ``run_backtest`` return the full flat
    record list rather than the reporting subset -- that is the documented
    behaviour of its ``explicit_years`` branch, and the per-(year, mode) matrix
    is what every correction here consumes.
    """
    from scripts.mc_pool_backtest import StrategiesFitter, run_backtest

    kwargs = dict(CANONICAL)
    kwargs.update(overrides)
    return run_backtest(
        years=list(years),
        hparam_fitter=StrategiesFitter(tuple(modes)),
        workers=workers,
        save_brackets=False,  # see the module docstring. Never change this.
        **kwargs,
    )


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)
    print(f"  [artifact] {path}")


def _provenance() -> Dict[str, Any]:
    """Run context, so an archived artifact can be tied back to the code."""
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "canonical_contract": dict(CANONICAL),
    }


# ---------------------------------------------------------------------------
# --multiplicity
# ---------------------------------------------------------------------------


def run_multiplicity(
    workers: int,
    n_resamples: int,
    out_dir: Path,
    reuse_matrix: Optional[Path] = None,
) -> Dict[str, Any]:
    """Correct the headline for having been selected as best-of-``ALL_MODES``.

    Every pre-existing multi-mode run in ``artifacts/backtest_runs/`` predates
    ``b73d351`` (2026-09-06, the play-in resolution fix), so its brackets
    contained teams that never played the Round of 64 and its P(1st) values are
    void rather than merely stale. There is nothing to reuse; the matrix has to
    be measured fresh.
    """
    from scripts.mc_pool_backtest import ALL_MODES, EVALUATION_YEARS
    from src.evaluation.multiplicity import paired_matrix_from_results, romano_wolf_stepdown

    modes = list(ALL_MODES)
    if BASELINE_MODE not in modes:
        modes.append(BASELINE_MODE)
    years = list(EVALUATION_YEARS)

    if reuse_matrix is not None:
        # Re-derive the correction from a stored matrix without re-simulating.
        # The 79-mode run is ~85 minutes; the stepdown on its output is
        # seconds, so any change to the CORRECTION must not require repeating
        # the MEASUREMENT. The stored per_year_per_mode block is the full
        # matrix, so this is lossless for everything downstream of it.
        with open(reuse_matrix) as f:
            stored = json.load(f)
        results = stored["per_year_per_mode"]
        elapsed = 0.0
        print(f"[multiplicity] reusing stored matrix from {reuse_matrix} ({len(results)} records)")
        print(f"[multiplicity] originally measured {stored.get('provenance', {}).get('timestamp')}")
    else:
        print(f"[multiplicity] {len(modes)} modes x {len(years)} seasons, workers={workers}")
        print(f"[multiplicity] seasons: {years}")
        t0 = time.time()
        results = _canonical_run(modes, years, workers)
        elapsed = time.time() - t0
        print(f"[multiplicity] backtest complete in {elapsed / 60:.1f} min, {len(results)} records")

    # The headline mode must cover every season or the correction is
    # meaningless -- require_modes makes that a hard failure rather than a
    # quietly smaller family.
    names, seasons, diffs = paired_matrix_from_results(
        results,
        baseline_mode=BASELINE_MODE,
        metric="p_first",
        require_modes=[HEADLINE_MODE],
    )
    print(f"[multiplicity] paired matrix: {len(names)} modes x {len(seasons)} seasons")
    dropped = sorted(set(modes) - set(names) - {BASELINE_MODE})
    if dropped:
        print(f"[multiplicity] dropped (incomplete season coverage): {dropped}")

    stepdown = romano_wolf_stepdown(diffs, names, n_resamples=n_resamples)

    # Aggregate P(1st) per mode, season-averaged -- the headline's own
    # definition (report_backtest_results averages p_first over seasons).
    by_mode: Dict[str, List[float]] = {}
    for rec in results:
        by_mode.setdefault(str(rec["mode"]), []).append(float(rec["p_first"]))
    aggregate = {m: sum(v) / len(v) for m, v in by_mode.items()}

    # "Best" is ambiguous and the two senses can disagree, so report both.
    # stepdown.best is the largest STUDENTIZED statistic -- the mode whose
    # advantage over the baseline is most reliable across seasons. That is the
    # right quantity for a hypothesis test, but it is not the same as the
    # largest raw aggregate P(1st), which is what the headline quotes. Naming
    # only one of them invites reading the other into it.
    contenders = {m: v for m, v in aggregate.items() if m != BASELINE_MODE}
    best_by_mean = max(contenders, key=lambda m: contenders[m])
    headline_mean = aggregate.get(HEADLINE_MODE, float("nan"))
    mean_gap_pp = (contenders[best_by_mean] - headline_mean) * 100.0
    n_surviving = sum(1 for s in stepdown.p_adjusted.values() if s < 0.05)

    # The same ranking with circular modes set aside -- the comparison a reader
    # actually wants when asking "does anything really beat the search?".
    non_circular = {m: v for m, v in contenders.items() if m not in CIRCULAR_MODES}
    best_non_circular = max(non_circular, key=lambda m: non_circular[m]) if non_circular else None

    p_max = stepdown.p_max_statistic
    verdict = (
        "survives_correction"
        if p_max < 0.05
        else ("indistinguishable_from_selection_noise" if p_max <= 0.20 else "substantially_a_selection_artifact")
    )

    payload = {
        "provenance": _provenance(),
        "measurement": "multiplicity correction over the surviving strategy family",
        "n_modes_run": len(modes),
        "n_modes_in_family": len(names),
        "seasons": seasons,
        "dropped_modes": dropped,
        "aggregate_p_first": aggregate,
        "baseline_mode": BASELINE_MODE,
        "headline_mode": HEADLINE_MODE,
        "headline_aggregate_p_first": aggregate.get(HEADLINE_MODE),
        "baseline_aggregate_p_first": aggregate.get(BASELINE_MODE),
        "best_mode_by_test_statistic": stepdown.best,
        "best_mode_by_aggregate_p_first": best_by_mean,
        "best_aggregate_p_first": contenders[best_by_mean],
        "headline_is_highest_scoring": best_by_mean == HEADLINE_MODE,
        "mean_gap_to_best_pp": round(mean_gap_pp, 3),
        "best_mode_is_circular": best_by_mean in CIRCULAR_MODES,
        "circular_modes": {m: why for m, why in CIRCULAR_MODES.items() if m in contenders},
        "best_non_circular_mode": best_non_circular,
        "best_non_circular_p_first": non_circular.get(best_non_circular) if best_non_circular else None,
        "n_modes_surviving_fwer_05": n_surviving,
        "stepdown": stepdown.to_dict(),
        "headline_p_adjusted": stepdown.p_adjusted.get(HEADLINE_MODE),
        "headline_p_unadjusted": stepdown.p_unadjusted.get(HEADLINE_MODE),
        "p_max_statistic": p_max,
        "verdict": verdict,
        "decision_rule": "pre-registered in the module docstring before this ran",
        "residual_not_corrected": (
            "This corrects over the strategy family that still exists in ALL_MODES. It does "
            "NOT correct over the search's history: candidate families that were built, "
            "measured, found worse and deleted (scripts/mc_pool_backtest.py:4026-4042 records "
            "two, with dates 2026-05-03 and 2026-05-16 and the numbers that justified each "
            "removal). Those specifications are gone from the code, so no resampling scheme "
            "can restore them to the family. Carried as removed_after_measuring entries in "
            "src.governance.pool_rdof_audit.REGISTRY."
        ),
        "wall_time_minutes": round(elapsed / 60, 1),
        "per_year_per_mode": [
            {"year": int(r["year"]), "mode": str(r["mode"]), "p_first": float(r["p_first"])} for r in results
        ],
    }
    _write(out_dir / "mode_multiplicity_2011_2025.json", payload)

    print(f"\n[multiplicity] {HEADLINE_MODE}: aggregate P(1st) = {headline_mean:.4f}")
    print(f"[multiplicity] best by test statistic:  {stepdown.best}")
    print(
        f"[multiplicity] best by aggregate P(1st): {best_by_mean} "
        f"({contenders[best_by_mean]:.4f}, {mean_gap_pp:+.2f}pp vs headline)"
    )
    print(f"[multiplicity] modes beating seed at FWER<0.05: {n_surviving}/{len(names)}")
    print(f"[multiplicity] p unadjusted            = {stepdown.p_unadjusted.get(HEADLINE_MODE)}")
    print(f"[multiplicity] p stepdown-adjusted     = {stepdown.p_adjusted.get(HEADLINE_MODE)}")
    print(f"[multiplicity] p max-statistic (best-of-family) = {p_max}")
    print(f"[multiplicity] VERDICT: {verdict}")
    return payload


# ---------------------------------------------------------------------------
# --sweep
# ---------------------------------------------------------------------------

# The specification curve. Each entry is one knob and the grid it is swept
# over; cells are one-at-a-time from the production setting, not a full
# factorial -- a factorial over four knobs is ~200 cells and several days, and
# the question here is "does any single unjustified knob move the headline",
# which one-at-a-time answers.
#
# Only knobs that are (a) tier 3 (freely tuned), (b) genuinely un-provenanced
# or provenanced by their own effect on this metric, and (c) outside the frozen
# 2027 spec. Nothing swept here is read by frozen_spec.capture_live_spec(), so
# the drift gate cannot be disturbed by running this.
SWEEP_AXES: Dict[str, Dict[str, Any]] = {
    "referee_noise_std": {
        "values": [0.08, 0.16, 0.24],
        "production": 0.16,
        "why": (
            "Cited to Lopez & Matthews (2015), a point-spread paper, and never fit to this "
            "repo's data -- audit finding M2. The one attempt to fit it "
            "(src/simulation/mc_calibration.py) was broken and was retired rather than fixed."
        ),
    },
    "pa_trials": {
        "values": [200, 500, 1000],
        "production": 500,
        "why": (
            "Raised 200 -> 500 BECAUSE IT HELPED on these same evaluation seasons "
            "(FINDINGS.md:137-139) -- provenance by its own effect on the metric being "
            "reported, which is the definition of a researcher degree of freedom."
        ),
    },
    "poolaware_risk_levels": {
        "values": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.2, 0.5, 0.8], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
        "labels": ["production", "coarse", "fine"],
        "production": [0.1, 0.3, 0.5, 0.7, 0.9],
        "why": "No numeric provenance recorded anywhere for this grid (poolaware_recipe.py:30).",
    },
    "poolaware_base_order": {
        "values": ["recipe", "reversed"],
        "production": "recipe",
        "why": (
            "The selector's tie-break is strictly-greater, so the recipe's base order can "
            "decide which bracket ships without changing any probability -- the recipe says "
            "so itself (poolaware_recipe.py:70-73). An undocumented DoF worth its own axis."
        ),
    },
}

# Sensitivity thresholds, in percentage points of aggregate P(1st). 1.5pp is
# one season-level standard error on the headline (audit H1), so it is the
# finest distinction the data supports.
_FLAT_PP = 1.5
_MILD_PP = 3.0


def _effective_dof(spread_pp: float) -> tuple[float, str]:
    """Flatness-discounted DoF for one knob.

    Kept from the deleted src/ml/evaluation/rdof_audit.py, along with its own
    honesty label: this is a heuristic, not a formal statistical quantity. Its
    point is that a knob whose whole grid gives the same answer did not
    actually cost a degree of freedom, and counting it as one overstates the
    problem as much as ignoring it understates it.
    """
    if spread_pp < _FLAT_PP:
        return 0.0, "flat_plateau"
    if spread_pp < _MILD_PP:
        return 0.5, "mild_slope"
    return 1.0, "sharp_peak"


def run_sweep(workers: int, out_dir: Path, axes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Specification curve over the un-provenanced knobs, on 2011-2025 only.

    Never reads 2026 or 2027. 2026 is the parameter-clean holdout this sweep
    exists to make valid; 2027 is sequestered. Deriving a parameter on a season
    and then evaluating that season is the contamination this whole item is
    about, so the year list is asserted, not assumed.
    """
    from scripts.mc_pool_backtest import EVALUATION_YEARS
    from src.governance.pool_rdof_audit import assert_not_sequestered

    years = list(EVALUATION_YEARS)
    assert_not_sequestered(years, context="RDoF specification curve")
    if 2026 in years:
        raise ValueError(
            "2026 is in the sweep years. It is the parameter-clean holdout; deriving "
            "parameters on it would destroy the only holdout status it can still have."
        )

    selected = list(axes) if axes else list(SWEEP_AXES)
    unknown = sorted(set(selected) - set(SWEEP_AXES))
    if unknown:
        raise ValueError(f"unknown sweep axis/axes {unknown}; known: {sorted(SWEEP_AXES)}")

    modes = [HEADLINE_MODE, BASELINE_MODE]
    per_axis: Dict[str, Any] = {}
    total_cells = sum(len(SWEEP_AXES[a]["values"]) for a in selected)
    print(f"[sweep] {len(selected)} axes, {total_cells} cells, {len(years)} seasons, workers={workers}")

    # Every axis's production cell is the SAME configuration -- one-at-a-time
    # sweeps all start from production -- so it is computed once and reused.
    # Without this, 4 of 11 cells are byte-identical re-runs of the baseline.
    cache: Dict[str, Dict[str, Any]] = {}

    def _cell(axis: str, value: Any, label: str, is_production: bool, index: int) -> Dict[str, Any]:
        key = "PRODUCTION" if is_production else f"{axis}={value!r}"
        if key in cache:
            print(f"[sweep] cell {index}/{total_cells}: {axis}={label} — reusing the production cell")
            return {**cache[key], "value": value, "label": label, "is_production": is_production, "reused": True}
        print(f"\n[sweep] cell {index}/{total_cells}: {axis}={label}")
        t0 = time.time()
        results = _canonical_run(modes, years, workers, **{axis: value})
        by_mode: Dict[str, List[float]] = {}
        for rec in results:
            by_mode.setdefault(str(rec["mode"]), []).append(float(rec["p_first"]))
        agg = {m: sum(v) / len(v) for m, v in by_mode.items()}
        headline = agg.get(HEADLINE_MODE)
        elapsed = (time.time() - t0) / 60
        print(
            f"[sweep] cell {index}/{total_cells} done in {elapsed:.1f} min: "
            f"P(1st)={headline:.4f} vs seed {agg.get(BASELINE_MODE, float('nan')):.4f}"
        )
        cell = {
            "value": value,
            "label": label,
            "is_production": is_production,
            "aggregate_p_first": headline,
            "baseline_aggregate_p_first": agg.get(BASELINE_MODE),
            "per_year": {str(int(r["year"])): float(r["p_first"]) for r in results if r["mode"] == HEADLINE_MODE},
            "wall_time_minutes": round(elapsed, 1),
            "reused": False,
        }
        cache[key] = cell
        return cell

    cell_index = 0
    for axis in selected:
        spec = SWEEP_AXES[axis]
        values = spec["values"]
        labels = spec.get("labels") or [str(v) for v in values]
        cells = []
        for value, label in zip(values, labels):
            cell_index += 1
            cells.append(_cell(axis, value, label, value == spec["production"], cell_index))

        vals = [c["aggregate_p_first"] for c in cells if c["aggregate_p_first"] is not None]
        spread_pp = (max(vals) - min(vals)) * 100.0 if len(vals) > 1 else 0.0
        eff_dof, category = _effective_dof(spread_pp)
        prod_cell = next((c for c in cells if c["is_production"]), None)
        # `or -1.0` would be wrong here: a cell that genuinely scored 0.0 is a
        # real measurement, not a missing one, and must not be ranked below a
        # failed cell.
        best_cell = max(cells, key=lambda c: -1.0 if c["aggregate_p_first"] is None else c["aggregate_p_first"])
        prod_p = prod_cell["aggregate_p_first"] if prod_cell else None
        best_p = best_cell["aggregate_p_first"]
        # Tie-aware. `max()` returns the FIRST maximum, so on a flat plateau
        # whichever value happens to sit earliest in the grid would be crowned
        # and production would read as suboptimal purely from ordering. Report
        # the gap, and treat a tie as production being best -- which on a flat
        # plateau is the substantive truth anyway.
        gap_pp = None if (prod_p is None or best_p is None) else (best_p - prod_p) * 100.0
        per_axis[axis] = {
            "why_swept": spec["why"],
            "cells": cells,
            "spread_pp": round(spread_pp, 2),
            "category": category,
            "effective_dof": eff_dof,
            "production_value": spec["production"],
            "production_p_first": prod_p,
            "best_value": best_cell["value"],
            "best_p_first": best_p,
            "gap_to_best_pp": None if gap_pp is None else round(gap_pp, 2),
            "production_is_best": bool(gap_pp is not None and gap_pp <= 1e-9),
        }
        print(f"[sweep] {axis}: spread {spread_pp:.2f}pp -> {category} ({eff_dof} effective DoF)")

    payload = {
        "provenance": _provenance(),
        "measurement": "one-at-a-time specification curve over un-provenanced tier-3 knobs",
        "seasons": years,
        "n_seasons": len(years),
        "excluded_seasons": {
            "2026": "parameter-clean holdout -- deriving parameters here would contaminate it",
            "2027": "sequestered prospective holdout",
        },
        "flat_threshold_pp": _FLAT_PP,
        "mild_threshold_pp": _MILD_PP,
        "threshold_basis": "1.5pp is one season-level standard error on the headline (audit H1)",
        "axes": per_axis,
        "total_effective_dof": round(sum(a["effective_dof"] for a in per_axis.values()), 2),
        "design_note": (
            "One-at-a-time from the production setting, not a full factorial: a factorial over "
            "these four axes is ~200 cells and several days, and the question asked here is "
            "whether any single unjustified knob moves the headline. Interaction effects are "
            "therefore NOT measured and this cannot rule them out."
        ),
        "not_adopted": (
            "No sweep optimum is adopted into the code by this script, deliberately. Adopting "
            "the best cell would itself be a new researcher degree of freedom -- exactly what "
            "the deleted rdof_audit.adopt_sensitivity_optima() did wrong. Any adoption is a "
            "separate decision requiring a SPEC_VERSION bump."
        ),
    }
    _write(out_dir / "specification_curve_2011_2025.json", payload)
    return payload


# ---------------------------------------------------------------------------
# --holdout-2026
# ---------------------------------------------------------------------------


def run_holdout_2026(workers: int, out_dir: Path) -> Dict[str, Any]:
    """Evaluate the parameter-clean (Level 2.5) holdout once, and lock it."""
    from src.governance.pool_rdof_audit import (
        PARAMETER_CLEAN_HOLDOUT,
        record_holdout_evaluation,
        registry_hash,
    )

    year = PARAMETER_CLEAN_HOLDOUT
    print(f"[holdout] evaluating {year} at production settings, workers={workers}")
    t0 = time.time()
    results = _canonical_run([HEADLINE_MODE, BASELINE_MODE], [year], workers)
    by_mode = {str(r["mode"]): float(r["p_first"]) for r in results}

    payload = {
        "provenance": _provenance(),
        "measurement": f"parameter-clean holdout evaluation, {year}",
        "evidence_level": "2.5 -- structurally contaminated, parameter-clean",
        "evidence_level_explanation": (
            f"{year} is NOT a Level-1 holdout and cannot be made one. The candidate-family set "
            "was pruned using aggregates that included it: scripts/mc_pool_backtest.py:4026-4042 "
            "records family (d) removed 2026-05-03 ('selected in only 1/15 years') and family "
            "(e) removed 2026-05-16, where a 15-season window is BACKTEST_YEARS = 2011-2026 "
            "excluding 2020. The 2026 tournament ended in April 2026, so its outcome was known. "
            "CONTAMINATED_EVAL_YEARS, which now strips 2026 from aggregates, was not added until "
            "2026-09-09 (a3fb412) -- four months after those prunings. The removed families no "
            "longer exist as code, so those decisions cannot be re-run on a 2026-free window. "
            "What IS true is that the prediction model never trains on it (train_noseed_model's "
            "walk-forward filter is y < max_year) and the parametric knobs are derived on "
            "2011-2025 only. That makes it parameter-clean, not untouched."
        ),
        "year": year,
        "p_first": by_mode,
        "registry_hash_at_evaluation": registry_hash(),
        "wall_time_minutes": round((time.time() - t0) / 60, 1),
    }
    _write(out_dir / f"holdout_{year}_parameter_clean.json", payload)
    record_holdout_evaluation(
        year=year,
        evidence_level="2.5",
        summary={"p_first": by_mode},
    )
    return payload


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def build_report(out_dir: Path) -> None:
    """Assemble the registry + whatever measurements are on disk into a report."""
    from src.governance.pool_rdof_audit import PoolRDOFReport

    def _load(name: str) -> Optional[Dict[str, Any]]:
        path = out_dir / name
        if not path.exists():
            print(f"  [report] {name} absent — section omitted")
            return None
        with open(path) as f:
            return json.load(f)

    report = PoolRDOFReport(
        multiplicity=_load("mode_multiplicity_2011_2025.json"),
        sensitivity=_load("specification_curve_2011_2025.json"),
        holdout=_load("holdout_2026_parameter_clean.json"),
    )
    text = report.to_text()
    print(text)
    _write(out_dir / "pool_rdof_audit.json", report.to_dict())
    txt_path = out_dir / "pool_rdof_audit.txt"
    txt_path.write_text(text)
    print(f"  [artifact] {txt_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-only", action="store_true", help="Registry + drift report only. No simulation.")
    parser.add_argument("--multiplicity", action="store_true", help="Full-family run + Romano-Wolf stepdown (slow).")
    parser.add_argument("--sweep", action="store_true", help="Specification curve over un-provenanced knobs (slow).")
    parser.add_argument("--holdout-2026", action="store_true", help="Evaluate the parameter-clean holdout once.")
    parser.add_argument(
        "--axes",
        nargs="+",
        default=None,
        help=f"Restrict --sweep to these axes. Known: {', '.join(sorted(SWEEP_AXES))}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RDOF_WORKERS", "8")),
        help="Year-level parallelism passed to run_backtest (not CLI-exposed on mc_pool_backtest).",
    )
    parser.add_argument("--n-resamples", type=int, default=10_000, help="Sign-flip draws for the stepdown.")
    parser.add_argument(
        "--reuse-matrix",
        type=Path,
        default=None,
        help="Re-derive the multiplicity correction from a stored matrix JSON instead of "
        "re-running the 79-mode backtest. Changing the CORRECTION must not require "
        "repeating the 85-minute MEASUREMENT.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not any([args.registry_only, args.multiplicity, args.sweep, args.holdout_2026]):
        parser.error("pick at least one of --registry-only / --multiplicity / --sweep / --holdout-2026")

    if args.multiplicity:
        run_multiplicity(args.workers, args.n_resamples, args.out_dir, reuse_matrix=args.reuse_matrix)
    if args.sweep:
        run_sweep(args.workers, args.out_dir, args.axes)
    if args.holdout_2026:
        run_holdout_2026(args.workers, args.out_dir)

    # The report is always rebuilt: it is the human-readable face of whatever
    # measurements exist, and a partial set must still render.
    build_report(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
