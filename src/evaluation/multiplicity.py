"""Family-wise multiplicity correction for a best-of-many strategy search.

WHY THIS EXISTS
---------------
The pool-strategy headline is the aggregate P(1st) of ONE mode --
``meta_region_poolaware`` -- which was chosen as the best of 74 candidate modes
(``scripts.mc_pool_backtest.ALL_MODES``) measured on the same 14 seasons. Audit
finding H2 ("Garden of forking paths") is that no correction was ever applied
for that choice: a maximum over 74 correlated estimates is biased upward even
when every mode is worthless, and the existing machinery does not measure it.

What was already here and why it is not enough:

  * ``report_backtest_results`` Bonferroni-corrects over modes, but on MEAN RANK,
    not P(1st) -- and Bonferroni over 74 highly correlated modes (they share
    opponent draws per repeat by design, see ``draw_selection_trials``) is so
    conservative as to be uninformative.
  * ``scripts.run_experiment._paired_permutation_on_metric`` sign-flips paired
    season deltas for ONE mode at a time and Bonferroni-adjusts afterwards.
    Same conservatism, and it never forms the max statistic.

Romano-Wolf stepdown fixes both problems at once. It resamples all modes under
ONE shared sign vector per draw, so the cross-mode correlation structure is
carried into the null rather than assumed away, and it steps down through the
ordered statistics so a strong winner does not have to pay for 73 also-rans.

WHAT IT CANNOT DO
-----------------
This corrects over the modes that STILL EXIST. It cannot correct over the
*history* of the search -- candidate families that were built, measured, found
worse, and deleted (``scripts/mc_pool_backtest.py:4026-4042`` records two, with
dates and numbers). Those specifications are gone from the code, so no
resampling scheme can put them back in the family. That residual is carried as
``removed_after_measuring`` entries in
``src.governance.pool_rdof_audit.REGISTRY`` and must be stated alongside any
adjusted p-value from here.

REFERENCE
---------
Romano, J. P. & Wolf, M. (2005), "Exact and Approximate Stepdown Methods for
Multiple Hypothesis Testing", JASA 100(469). The studentized, recentered
variant is used here, as recommended there for heterogeneous variances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# Resample count. 10_000 matches the existing permutation test in
# scripts/run_experiment.py:808 so the two report comparable resolution; the
# smallest p-value representable is 1/(1+10_000) ~= 1e-4.
DEFAULT_N_RESAMPLES = 10_000

# Fixed so a re-run reproduces the published number. Same value as the
# existing permutation test, for the same reason.
DEFAULT_SEED = 42

# Chunk size for the resampling loop. (B, M, n) fully vectorized at B=10_000,
# M=74, n=14 is ~83MB of float64; chunking keeps peak memory flat without
# giving up vectorization.
_CHUNK = 500


def _studentized(d: np.ndarray) -> np.ndarray:
    """One-sample t statistic per row of ``d`` (modes x seasons).

    Studentized rather than raw-mean because the modes have visibly different
    season-to-season variance -- a deterministic ``fixed_*`` mode and a
    stochastic ``meta_*`` mode are not on the same scale, and Romano-Wolf's
    validity under heterogeneity is what motivates studentizing.
    """
    n = d.shape[1]
    mean = d.mean(axis=1)
    sd = d.std(axis=1, ddof=1)
    # A degenerate row (identical every season) has no sampling variation. Its
    # t is 0 if the mean is 0 and otherwise unbounded; clamp to a large finite
    # value so it sorts correctly without producing inf/nan downstream.
    out = np.zeros_like(mean)
    ok = sd > 0
    out[ok] = mean[ok] / (sd[ok] / np.sqrt(n))
    degenerate = (~ok) & (mean != 0)
    out[degenerate] = np.sign(mean[degenerate]) * 1e6
    return out


@dataclass(frozen=True)
class StepdownResult:
    """Outcome of a Romano-Wolf stepdown over a family of strategies.

    Attributes:
        names: Strategy names, in the order they were passed in.
        n_seasons: Seasons per strategy -- the unit of independence. NOT
            bracket-repeats: audit finding H1 established that repeats within a
            season are not independent observations, and a CI over them
            understates by ~3x.
        mean_diff: Mean paired difference vs the baseline, per strategy.
        t_stat: Studentized statistic per strategy.
        p_unadjusted: One-at-a-time sign-flip p-value, no correction. Reported
            only so the size of the correction is visible.
        p_adjusted: Family-wise stepdown-adjusted p-value per strategy.
        p_max_statistic: The single-step p-value for the family's best
            performer -- P(max over ALL strategies of the null statistic >=
            the observed best). This is the number that answers "is the
            headline just the best of many?".
        best: Name of the strategy with the largest observed statistic.
        n_resamples: Sign-flip draws used.
    """

    names: List[str]
    n_seasons: int
    mean_diff: Dict[str, float]
    t_stat: Dict[str, float]
    p_unadjusted: Dict[str, float]
    p_adjusted: Dict[str, float]
    p_max_statistic: float
    best: str
    n_resamples: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_seasons": self.n_seasons,
            "n_resamples": self.n_resamples,
            "n_strategies": len(self.names),
            "best": self.best,
            "p_max_statistic": self.p_max_statistic,
            "per_strategy": {
                name: {
                    "mean_diff": self.mean_diff[name],
                    "t_stat": self.t_stat[name],
                    "p_unadjusted": self.p_unadjusted[name],
                    "p_adjusted": self.p_adjusted[name],
                }
                for name in self.names
            },
        }


def romano_wolf_stepdown(
    diffs: Sequence[Sequence[float]],
    names: Sequence[str],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> StepdownResult:
    """Stepdown-adjusted one-sided p-values for ``strategy > baseline``.

    Args:
        diffs: ``(n_strategies, n_seasons)`` paired differences, strategy minus
            baseline, one column per season. Every strategy must be measured on
            the SAME seasons in the SAME order -- the pairing is what licenses
            sign-flipping, and a ragged family silently breaks it.
        names: Strategy names, parallel to ``diffs`` rows.
        n_resamples: Sign-flip draws.
        seed: RNG seed, fixed for reproducibility.

    Returns:
        A :class:`StepdownResult`.

    Raises:
        ValueError: if the family is empty, ragged, or has fewer than 2 seasons
            (a sign-flip null over one season has two possible values and
            cannot produce a meaningful p-value).
    """
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 2 or d.shape[0] == 0:
        raise ValueError(f"diffs must be a non-empty 2-D (strategies x seasons) array, got shape {d.shape}")
    if len(names) != d.shape[0]:
        raise ValueError(f"names has {len(names)} entries but diffs has {d.shape[0]} rows")
    n_strategies, n_seasons = d.shape
    if n_seasons < 2:
        raise ValueError(
            f"need at least 2 seasons for a sign-flip null, got {n_seasons}. "
            "The season is the unit of independence here (audit H1); repeats "
            "within a season cannot substitute."
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("diffs contains non-finite values; a ragged or partially-failed run cannot be corrected")

    t_obs = _studentized(d)

    # Recenter before resampling. Under the stepdown, the null being simulated
    # at each step is "every strategy still in the remaining set has zero mean
    # effect". Sign-flipping the RAW differences would be the right null only
    # if that were true of the observed data too; recentering makes it true by
    # construction, which is the studentized-and-recentered variant Romano-Wolf
    # recommends. (For the single-step max statistic at the first step the two
    # agree closely; they diverge once strong winners are peeled off.)
    d_centered = d - d.mean(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    t_null = np.empty((n_resamples, n_strategies), dtype=float)
    for start in range(0, n_resamples, _CHUNK):
        stop = min(start + _CHUNK, n_resamples)
        # One sign vector per draw, SHARED across strategies -- this is what
        # carries the cross-mode correlation into the null. Drawing per
        # strategy would destroy it and reduce to something Bonferroni-like.
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, 1, n_seasons))
        flipped = d_centered[None, :, :] * signs  # (chunk, strategies, seasons)
        mean = flipped.mean(axis=2)
        sd = flipped.std(axis=2, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sd > 0, mean / (sd / np.sqrt(n_seasons)), 0.0)
        t_null[start:stop] = t

    # One-at-a-time p-values, for showing the size of the correction only.
    # (1 + count) / (1 + B) is the standard add-one convention: it keeps the
    # p-value strictly positive and the test exact-valid.
    p_unadj = (1.0 + np.sum(t_null >= t_obs[None, :], axis=0)) / (1.0 + n_resamples)

    # Stepdown: walk the observed statistics from largest to smallest, each
    # time forming the max over only the strategies not yet rejected, then
    # enforce monotonicity so an adjusted p can never decrease down the order.
    order = np.argsort(-t_obs)
    p_adj = np.empty(n_strategies, dtype=float)
    running = 0.0
    p_max_statistic = float("nan")
    for position, idx in enumerate(order):
        remaining = order[position:]
        max_null = t_null[:, remaining].max(axis=1)
        p = float((1.0 + np.sum(max_null >= t_obs[idx])) / (1.0 + n_resamples))
        if position == 0:
            # Step 0's max is over the WHOLE family, so this is the
            # single-step, best-of-all-strategies p-value.
            p_max_statistic = p
        running = max(running, p)
        p_adj[idx] = running

    name_list = list(names)
    return StepdownResult(
        names=name_list,
        n_seasons=n_seasons,
        mean_diff={n: float(v) for n, v in zip(name_list, d.mean(axis=1))},
        t_stat={n: float(v) for n, v in zip(name_list, t_obs)},
        p_unadjusted={n: float(v) for n, v in zip(name_list, p_unadj)},
        p_adjusted={n: float(v) for n, v in zip(name_list, p_adj)},
        p_max_statistic=p_max_statistic,
        best=name_list[int(order[0])],
        n_resamples=n_resamples,
    )


def paired_matrix_from_results(
    results: Sequence[Dict[str, object]],
    baseline_mode: str,
    metric: str = "p_first",
    require_modes: Optional[Sequence[str]] = None,
) -> tuple[List[str], List[int], np.ndarray]:
    """Reshape ``run_backtest`` records into a paired (modes x seasons) matrix.

    ``run_backtest`` returns a flat list of one dict per (year, mode) with keys
    ``year`` / ``mode`` / ``p_first`` / ... (see
    ``scripts/mc_pool_backtest.py:4500``). The stepdown needs a rectangular,
    baseline-subtracted matrix over the seasons where EVERY mode reported.

    Modes are dropped, loudly, rather than padded: a mode present for 9 of 14
    seasons is not comparable to one present for all 14, and silently
    imputing would corrupt the pairing that licenses sign-flipping.

    Args:
        results: Flat ``run_backtest`` output.
        baseline_mode: Mode to subtract, e.g. ``"seed"``.
        metric: Record key to difference.
        require_modes: If given, raise when any of these is not present on
            every common season. Use it for modes whose absence would
            invalidate the report (the headline mode, above all).

    Returns:
        ``(mode_names, seasons, diffs)`` where ``diffs`` is
        ``(len(mode_names), len(seasons))``, baseline excluded from the rows.

    Raises:
        ValueError: if the baseline is missing, nothing survives, or a
            ``require_modes`` entry does not cover the common seasons.
    """
    by_mode: Dict[str, Dict[int, float]] = {}
    for rec in results:
        mode = str(rec["mode"])
        by_mode.setdefault(mode, {})[int(rec["year"])] = float(rec[metric])  # type: ignore[index]

    if baseline_mode not in by_mode:
        raise ValueError(f"baseline mode {baseline_mode!r} absent from results; have {sorted(by_mode)[:10]}...")

    baseline = by_mode[baseline_mode]
    # Seasons common to the baseline and to every mode that covers the baseline
    # fully. Two passes: establish the candidate season set from the baseline,
    # then keep only modes that cover all of it.
    seasons = sorted(baseline)
    kept: List[str] = []
    for mode, per_year in by_mode.items():
        if mode == baseline_mode:
            continue
        if all(y in per_year for y in seasons):
            kept.append(mode)

    missing = sorted(set(require_modes or ()) - set(kept))
    if missing:
        raise ValueError(
            f"required mode(s) {missing} do not cover all {len(seasons)} baseline seasons; "
            "the run is incomplete and must not be corrected as if it were"
        )
    if not kept:
        raise ValueError("no mode covers every baseline season; nothing to correct")

    kept.sort()
    diffs = np.array([[by_mode[m][y] - baseline[y] for y in seasons] for m in kept], dtype=float)
    return kept, seasons, diffs
