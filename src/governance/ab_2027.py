"""Pre-registered 2027 A/B: does per-season selection earn its complexity?

WHY THIS EXISTS
---------------
Audit finding H11 established that the production strategy,
``meta_region_poolaware`` -- which generates ~20-25 candidate brackets per
season and picks one by simulated P(1st) -- is not clearly better than a fixed
rule that does none of that. Recommendation 16 asked for a pre-registered 2027
A/B to settle it, with a **non-circular** fixed arm: the mode that actually tops
the P(1st) table, ``fixed_blendA100_r35``, is built from the referee's own
probabilities and measures circularity rather than construction.

THE HEADLINE RESULT OF PRE-REGISTERING IT: IT CANNOT BE SETTLED
---------------------------------------------------------------
Doing the power calculation before writing the protocol -- rather than after
collecting data, which is when it usually gets done and is usually too late --
shows the proposed A/B is futile on the metric that matters. Measured over the
14 evaluation seasons:

    statistic     meta      fixed    paired diff   SD      p      n for 80% power
    P(1st)        .1200     .1100    +1.00pp      4.77pp  .447    181 seasons
    mean rank     10.52     9.14     -1.38 pos    2.12    .030     21 seasons

**181 seasons.** At one tournament a year that is the year 2207. Even the most
sensitive statistic needs 21. A single prospective season contributes about
1/181 of the evidence required, and no honest protocol can pretend otherwise.

So this module does NOT pre-register a test that will conclude. It pre-registers
an *accumulating ledger* with a stopping rule, for three reasons, none of which
is "we expect an answer soon":

  1. It makes the 2027 observation admissible. Without a rule fixed in advance,
     whatever happens in 2027 can be read either way after the fact -- and the
     temptation to read one season as settling this is exactly what audit
     finding H2 is about. A pre-registration that says "this is observation 1
     of ~181" is a defence against that reading, which is its main job.
  2. It records the arms precisely and hashes them, so "the fixed arm" cannot
     quietly become a different fixed arm later.
  3. The data accrues whether or not anyone is collecting it deliberately. A
     ledger costs nothing per season and is the only way the question is ever
     answerable at all.

THE TWO METRICS DISAGREE IN SIGN, AND THAT IS THE FINDING
----------------------------------------------------------
``meta_region_poolaware`` is nominally ahead on P(1st) (+1.00pp, in 6 of 14
seasons, p=.447) and clearly behind on mean rank (-1.38 positions, in 11 of 14
seasons, p=.030). Both can be true: a higher-variance bracket wins outright
more often while finishing worse on average. The production selector's own
comment says as much -- *"a bracket that beats 70% of opponents but never wins
outright scores high on rank but has low P(1st). Binary is the correct unbiased
estimator of what pays out."*

For a winner-take-all pool, P(1st) is the objective that matters, so the
production choice is defensible. But the selector is being credited with an
advantage that is not statistically distinguishable from zero on its own
objective (6 of 14 seasons), while it is distinguishably worse on placement.
Both metrics are therefore primary here, pre-declared, and neither may be
dropped after the fact for disagreeing.

WHAT 2027 IS AND IS NOT
-----------------------
2027 is the one season this project has that is genuinely untouched
(``pool_rdof_audit.SEQUESTERED_YEARS``). Spending it is a real cost, and it
should be spent on the question it can answer -- the headline claim of
``PROSPECTIVE_2027_v2.md`` -- not on a 1pp difference that needs two centuries.
This A/B rides along at no extra cost, because both brackets are produced by
code that already runs. It does not consume the holdout's evidentiary value for
the primary claim; it only requires that both brackets be built and recorded
before the tournament starts.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- The arms -------------------------------------------------------------

#: The production strategy: ~20-25 candidates per season, one chosen by
#: simulated P(1st). The thing whose complexity is under test.
ARM_TREATMENT = "meta_region_poolaware"

#: The control: one fixed rule, alpha=0.5 blend at risk 0.40, no per-season
#: candidate generation and no selection step.
#:
#: NOT ``fixed_blendA100_r35``, despite that mode topping the P(1st) table.
#: alpha=1.0 is pure seed probabilities and the referee is the same seed model,
#: so it is graded by its own source -- audit finding H11. Using it would
#: measure circularity and call the result a strategy comparison.
ARM_CONTROL = "fixed_blend_r40"

#: Explicitly barred as a control, with the reason, so the exclusion is a
#: recorded decision rather than an omission someone later "fixes".
BARRED_CONTROLS = {
    "fixed_blendA100_r35": (
        "alpha=1.0 is pure seed_rp and the referee is seed_pw: the arm would be graded by "
        "its own probability source. See audit finding H11."
    ),
}

#: Both are primary. Declared together, in advance, precisely because they
#: disagree in sign on the historical window -- dropping the inconvenient one
#: after seeing 2027 is the failure this pre-registration exists to prevent.
PRIMARY_METRICS = ("p_first", "mean_rank")

#: Sign convention: +1 means "larger is better for the treatment arm".
METRIC_DIRECTION = {"p_first": +1, "mean_rank": -1}

#: First prospective season. Sequestered; see pool_rdof_audit.SEQUESTERED_YEARS.
FIRST_SEASON = 2027

# --- What the historical window says, fixed here as the prior --------------
#
# Measured on the 14 evaluation seasons under the canonical contract, BEFORE
# this protocol was written. Recorded so that a later reader can check whether
# the prospective data is consistent with the retrospective estimate, and so
# the power numbers below cannot be quietly re-derived from a friendlier
# window.
HISTORICAL_ESTIMATE: Dict[str, Dict[str, float]] = {
    "p_first": {
        "treatment_mean": 0.1200,
        "control_mean": 0.1100,
        # RAW difference, treatment minus control. The direction multiplier in
        # METRIC_DIRECTION is applied separately; storing an already-signed
        # value here and multiplying again silently flipped mean_rank's sign,
        # making the control's advantage read as the treatment's.
        "paired_diff_raw": 0.0100,
        "paired_sd": 0.0477,
        "cohens_d": 0.210,
        "p_value": 0.447,
        "treatment_better_seasons": 6,
        "n_seasons": 14,
        "seasons_for_80pct_power": 181,
    },
    "mean_rank": {
        "treatment_mean": 10.5164,
        "control_mean": 9.1368,
        # +1.38 raw means the treatment's average finishing position is 1.38
        # places WORSE, because lower rank is better (METRIC_DIRECTION = -1).
        "paired_diff_raw": 1.3796,
        "paired_sd": 2.1212,
        "cohens_d": -0.650,
        "p_value": 0.030,
        "treatment_better_seasons": 3,
        "n_seasons": 14,
        "seasons_for_80pct_power": 21,
    },
}

#: Two-sided alpha for the eventual test.
ALPHA = 0.05

#: Where prospective observations accumulate. Inside the artifacts tree so it
#: is committed alongside the measurements it records, same reasoning as the
#: holdout lockfile.
LEDGER_PATH = Path("artifacts/headline_measurement/ab_2027_ledger.json")

#: Frozen spec, hashed, so the protocol cannot drift after data starts arriving.
FROZEN_SPEC_PATH = Path("configs/frozen/prospective_2027_ab.json")

SPEC_VERSION = "2027.ab.v1"
FREEZE_DATE = "2026-09-13"


class PreregistrationViolation(RuntimeError):
    """An action that would break the pre-registered protocol."""


def spec() -> Dict[str, Any]:
    """The protocol, as data. This is what gets hashed."""
    return {
        "spec_version": SPEC_VERSION,
        "freeze_date": FREEZE_DATE,
        "question": (
            "Does per-season candidate selection (meta_region_poolaware) beat a fixed rule "
            "(fixed_blend_r40) that does no per-season selection?"
        ),
        "arm_treatment": ARM_TREATMENT,
        "arm_control": ARM_CONTROL,
        "barred_controls": dict(BARRED_CONTROLS),
        "primary_metrics": list(PRIMARY_METRICS),
        "metric_direction": dict(METRIC_DIRECTION),
        "first_season": FIRST_SEASON,
        "alpha": ALPHA,
        "test": "two-sided paired t-test across seasons, both metrics, no multiplicity correction across the two because both are primary",
        "historical_estimate": HISTORICAL_ESTIMATE,
        "stopping_rule": stopping_rule(),
        "measurement_contract": "src/evaluation/canonical_contract.py",
        "honest_expectation": (
            "This will not conclude. The decision-relevant metric needs 181 seasons at 80% "
            "power and the most sensitive one needs 21. The ledger exists so the data is "
            "admissible and so no single season is mistaken for an answer, not because an "
            "answer is anticipated."
        ),
    }


def stopping_rule() -> Dict[str, Any]:
    """When, if ever, this concludes -- fixed in advance.

    No interim analyses and no early stopping. Peeking at an accumulating
    series and stopping when it looks significant inflates the error rate
    exactly the way the forking paths in audit finding H2 do, and at one
    observation per year the temptation to peek has decades to work on
    whoever maintains this.
    """
    return {
        "analyse_at_seasons": [21, 181],
        "rationale": (
            "21 is 80% power on mean_rank, 181 on p_first, from the historical effect sizes "
            "recorded above. Analysing at any other count is an interim look."
        ),
        "no_interim_analysis": True,
        "no_early_stopping": True,
        "per_season_reporting": (
            "Each season's numbers are recorded and may be reported as a running tally. A "
            "running tally is a description, not a test: no significance claim may be made "
            "before a scheduled analysis point."
        ),
        "if_effect_absent_at_21": (
            "If mean_rank has not resolved by 21 prospective seasons, the honest conclusion "
            "is that the difference is too small to matter at any horizon this project "
            "operates on, and the complexity question should be decided on grounds other "
            "than measured performance -- maintenance cost, explainability, or the "
            "in-sample evidence that already exists."
        ),
    }


def spec_hash() -> str:
    """SHA-256 over the canonical protocol."""
    blob = json.dumps(spec(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def freeze(path: Path = FROZEN_SPEC_PATH) -> Dict[str, Any]:
    """Write the frozen protocol. Refuses to overwrite a differing one.

    A pre-registration that can be rewritten is not one. Changing the protocol
    is permitted -- bumping ``SPEC_VERSION`` starts a new one and invalidates
    the old -- but silently editing it in place is not.
    """
    payload = dict(spec())
    payload["spec_hash"] = spec_hash()
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if existing.get("spec_hash") != payload["spec_hash"]:
            raise PreregistrationViolation(
                f"{path} already holds a different protocol (frozen "
                f"{existing.get('spec_hash', '?')[:12]}, live {payload['spec_hash'][:12]}). A "
                "pre-registration is not editable in place: bump SPEC_VERSION to start a new "
                "one, which invalidates the old one's claim, and say why in "
                "PROSPECTIVE_2027_AB.md."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def verify(path: Path = FROZEN_SPEC_PATH) -> Dict[str, Any]:
    """Compare the live protocol against the frozen one."""
    if not path.exists():
        return {"frozen": False, "matches": False, "reason": f"{path} does not exist; run --freeze"}
    with open(path) as f:
        frozen = json.load(f)
    live = spec_hash()
    drifted = []
    frozen_body = {k: v for k, v in frozen.items() if k != "spec_hash"}
    live_body = spec()
    for key in sorted(set(frozen_body) | set(live_body)):
        if frozen_body.get(key) != live_body.get(key):
            drifted.append(key)
    return {
        "frozen": True,
        "matches": frozen.get("spec_hash") == live,
        "frozen_hash": frozen.get("spec_hash"),
        "live_hash": live,
        "drifted_fields": drifted,
    }


# --- The accumulating ledger ----------------------------------------------


def _read_ledger() -> Dict[str, Any]:
    if not LEDGER_PATH.exists():
        return {"spec_version": SPEC_VERSION, "observations": []}
    try:
        with open(LEDGER_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"spec_version": SPEC_VERSION, "observations": []}
    data.setdefault("observations", [])
    return data


def record_season(year: int, treatment: Dict[str, float], control: Dict[str, float]) -> Dict[str, Any]:
    """Record one prospective season's result for both arms.

    Raises:
        PreregistrationViolation: if the season predates the protocol, if a
            primary metric is missing, or if the season is already recorded.
            Re-recording is barred because a season whose numbers can be
            revised is a season whose numbers can be chosen.
    """
    if year < FIRST_SEASON:
        raise PreregistrationViolation(
            f"{year} predates this pre-registration (first season {FIRST_SEASON}). Retrospective "
            "seasons belong in HISTORICAL_ESTIMATE, which was fixed before the protocol was "
            "written; adding them here would make the prospective series retrospective."
        )
    missing = [m for m in PRIMARY_METRICS if m not in treatment or m not in control]
    if missing:
        raise PreregistrationViolation(
            f"both arms must report every primary metric; missing {missing}. Both metrics were "
            "declared primary precisely because they disagree in sign, so reporting one is "
            "reporting the convenient half."
        )

    data = _read_ledger()
    if any(int(o["year"]) == int(year) for o in data["observations"]):
        raise PreregistrationViolation(
            f"{year} is already recorded. A pre-registered observation is written once; if the "
            "measurement was wrong, say so in the ledger's notes and bump SPEC_VERSION rather "
            "than overwriting it."
        )

    entry = {
        "year": int(year),
        "treatment": {m: float(treatment[m]) for m in PRIMARY_METRICS},
        "control": {m: float(control[m]) for m in PRIMARY_METRICS},
        "spec_hash": spec_hash(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    data["observations"].append(entry)
    data["observations"].sort(key=lambda o: o["year"])
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(LEDGER_PATH)
    return entry


def tally(observations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """A running description of the prospective series. NOT a test.

    Deliberately reports no p-value. At any season count below the scheduled
    analysis points a p-value is an interim look, and the whole reason for a
    stopping rule is that interim looks at a series accumulating one point a
    year will eventually produce a significant-looking result by chance.
    """
    obs = observations if observations is not None else _read_ledger()["observations"]
    n = len(obs)
    out: Dict[str, Any] = {
        "n_seasons": n,
        "seasons": [o["year"] for o in obs],
        "per_metric": {},
        "is_a_test": False,
        "next_scheduled_analysis": next(
            (k for k in stopping_rule()["analyse_at_seasons"] if k > n), None
        ),
    }
    for metric in PRIMARY_METRICS:
        direction = METRIC_DIRECTION[metric]
        diffs = [(o["treatment"][metric] - o["control"][metric]) * direction for o in obs]
        out["per_metric"][metric] = {
            "treatment_better_seasons": sum(1 for d in diffs if d > 0),
            "control_better_seasons": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
            "mean_signed_diff": (sum(diffs) / n) if n else None,
            "historical_expectation": HISTORICAL_ESTIMATE[metric]["paired_diff_raw"] * direction,
        }
    out["reminder"] = (
        f"{n} of the {HISTORICAL_ESTIMATE['p_first']['seasons_for_80pct_power']} seasons needed "
        "to resolve p_first at 80% power. No significance claim is licensed here."
    )
    return out
