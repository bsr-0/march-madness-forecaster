"""Cross-source statistical agreement testing for public pick data.

Before blindly averaging ESPN/Yahoo/CBS, this module tests whether sources
agree.  Detects and quarantines corrupted sources so they don't pollute
the consensus estimate.

Algorithm overview:
1. For each pair of sources, compute Spearman rank correlation on
   championship pick percentages across all shared teams.
2. For each critical round (CHAMP, F4, E8), repeat the pairwise check.
3. If any pairwise ρ < min_correlation for the CHAMP round, identify
   the outlier source (the one that disagrees with the other two).
4. Compute recommended_weights: start from configured weights, then
   down-weight flagged sources proportionally to their disagreement.

We use Spearman (not Pearson) because rank correlation is robust to
scale differences between sources (ESPN reports 0-100, a corrupted
source might report 0-1).  What matters for leverage is whether the
*ordering* of teams is consistent.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Avoid hard scipy dependency — fall back to a pure-Python Spearman
# implementation when scipy is not available.
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _rank(vals: List[float]) -> List[float]:
    """Assign average ranks to values (1-based), handling ties."""
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Uses scipy when available; otherwise a pure-Python fallback.
    Returns the correlation coefficient (float in [-1, 1]).
    """
    if _HAS_SCIPY:
        rho, _ = _scipy_spearmanr(x, y)
        return float(rho) if not math.isnan(rho) else 0.0

    # Pure-Python fallback: rank-transform then compute Pearson on ranks.
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    if n < 2:
        return 0.0
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _spearman_p_value(rho: float, n: int) -> float:
    """Approximate two-tailed p-value for Spearman ρ.

    Uses the t-distribution approximation: t = ρ * sqrt((n-2)/(1-ρ²))
    with n-2 degrees of freedom.  For small n this is the standard
    approach (Zar, *Biostatistical Analysis*, 5th ed., §19.6).

    Falls back to scipy.stats.t if available; otherwise uses a normal
    approximation (reasonable for df ≥ 10).
    """
    if n < 3:
        return 1.0
    if abs(rho) >= 1.0:
        return 0.0

    t_stat = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    df = n - 2

    if _HAS_SCIPY:
        from scipy.stats import t as t_dist
        return float(2.0 * t_dist.sf(abs(t_stat), df))

    # Normal approximation (conservative for small df).
    # Using the survival function of the standard normal.
    z = abs(t_stat)
    # Abramowitz & Stegun approximation 26.2.17
    p = 0.3275911
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429,
    )
    t_val = 1.0 / (1.0 + p * z)
    poly = ((((a5 * t_val + a4) * t_val + a3) * t_val + a2) * t_val + a1) * t_val
    one_tail = poly * math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    return max(0.0, min(1.0, 2.0 * one_tail))


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceAgreementReport:
    """Result of cross-source agreement analysis."""

    pairwise_rank_correlations: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )  # (source_a, source_b) -> {round: spearman_rho}
    flagged_sources: List[str] = field(default_factory=list)
    agreement_level: str = "high"  # "high", "moderate", "low", "conflicting"
    recommended_weights: Dict[str, float] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Round attribute mapping
# ---------------------------------------------------------------------------

# Maps round name → PublicPicks attribute name.
_ROUND_ATTR = {
    "R64": "round_of_64_pct",
    "R32": "round_of_32_pct",
    "S16": "sweet_16_pct",
    "E8": "elite_8_pct",
    "F4": "final_four_pct",
    "CHAMP": "champion_pct",
}


# ---------------------------------------------------------------------------
# Core agreement assessment
# ---------------------------------------------------------------------------

def assess_source_agreement(
    sources: Dict[str, "ConsensusData"],
    min_correlation: float = 0.85,
    significance_level: float = 0.05,
    critical_rounds: Tuple[str, ...] = ("CHAMP", "F4", "E8"),
    configured_weights: Optional[Dict[str, float]] = None,
) -> SourceAgreementReport:
    """Test whether multiple pick sources statistically agree.

    A source pair is considered "in agreement" when:
      1. Their Spearman ρ on CHAMP picks is ≥ *min_correlation*, **and**
      2. The correlation is statistically significant at *significance_level*
         (two-tailed t-test on ρ).

    Condition (2) prevents small-sample false confidence: with n=6
    shared teams, ρ=0.85 has p ≈ 0.03 and might pass, but ρ=0.70 with
    n=6 has p ≈ 0.12 and would not.

    Args:
        sources: Mapping of source name → ConsensusData.
        min_correlation: Spearman ρ threshold below which a source pair
            is considered in disagreement.
        significance_level: Two-tailed p-value threshold for the ρ
            significance test.  Default 0.05.
        critical_rounds: Rounds to focus on for agreement testing.
        configured_weights: Starting weights (e.g. {"espn": 0.5, ...}).
            If *None*, uses equal weights.

    Returns:
        SourceAgreementReport with diagnostics and recommended weights.
    """
    source_names = sorted(sources.keys())
    report = SourceAgreementReport()

    if len(source_names) < 2:
        # Cannot assess agreement with fewer than two sources.
        report.agreement_level = "unknown"
        report.details.append(
            f"Only {len(source_names)} source(s) available; skipping agreement check."
        )
        if configured_weights:
            report.recommended_weights = dict(configured_weights)
        elif source_names:
            report.recommended_weights = {source_names[0]: 1.0}
        return report

    # ------------------------------------------------------------------
    # 1. Pairwise rank correlations for each critical round
    # ------------------------------------------------------------------
    # Track sample sizes so the significance test in step 2 is valid.
    pair_n_shared: Dict[Tuple[str, str], int] = {}

    for i, src_a in enumerate(source_names):
        for src_b in source_names[i + 1:]:
            shared_teams = sorted(
                set(sources[src_a].teams.keys()) & set(sources[src_b].teams.keys())
            )
            # Minimum n=10 for Spearman ρ.  With fewer shared teams,
            # the t-distribution approximation (df = n-2) is unreliable
            # and p-values are meaningless.  At n=10, df=8, which is the
            # minimum for a stable t-distribution tail estimate.
            # Reference: Zar, *Biostatistical Analysis*, 5th ed., §19.6.
            _MIN_SHARED_TEAMS = 10
            if len(shared_teams) < _MIN_SHARED_TEAMS:
                report.details.append(
                    f"{src_a}-{src_b}: only {len(shared_teams)} shared teams "
                    f"(need ≥{_MIN_SHARED_TEAMS}), skipping."
                )
                continue

            pair_key = (src_a, src_b)
            pair_n_shared[pair_key] = len(shared_teams)
            report.pairwise_rank_correlations[pair_key] = {}

            for rnd in critical_rounds:
                attr = _ROUND_ATTR.get(rnd)
                if attr is None:
                    continue
                vals_a = [getattr(sources[src_a].teams[t], attr) for t in shared_teams]
                vals_b = [getattr(sources[src_b].teams[t], attr) for t in shared_teams]
                rho = _spearmanr(vals_a, vals_b)
                report.pairwise_rank_correlations[pair_key][rnd] = rho

    # ------------------------------------------------------------------
    # 2. Identify flagged sources (majority-voting on CHAMP round)
    # ------------------------------------------------------------------
    # A pair "agrees" iff ρ ≥ min_correlation AND the correlation is
    # statistically significant at the chosen level.  This prevents
    # small-sample false confidence (e.g. n=6, ρ=0.7, p=0.12).
    def _pair_agrees(pair_key: Tuple[str, str]) -> bool:
        rho = report.pairwise_rank_correlations.get(pair_key, {}).get("CHAMP")
        if rho is None or rho < min_correlation:
            return False
        n = pair_n_shared.get(pair_key, 0)
        p = _spearman_p_value(rho, n)
        return p < significance_level

    # Build a symmetric {(src_a, src_b): champ_rho} lookup.
    champ_rhos: Dict[Tuple[str, str], float] = {}
    for (src_a, src_b), rounds in report.pairwise_rank_correlations.items():
        rho = rounds.get("CHAMP")
        if rho is not None:
            champ_rhos[(src_a, src_b)] = rho
            champ_rhos[(src_b, src_a)] = rho  # symmetric

    # Per-source average ρ (used later for weight adjustment).
    source_avg_rho: Dict[str, float] = {s: 0.0 for s in source_names}
    source_pair_count: Dict[str, int] = {s: 0 for s in source_names}
    for (src_a, src_b), rounds in report.pairwise_rank_correlations.items():
        rho = rounds.get("CHAMP")
        if rho is None:
            continue
        source_avg_rho[src_a] += rho
        source_avg_rho[src_b] += rho
        source_pair_count[src_a] += 1
        source_pair_count[src_b] += 1
    for s in source_names:
        if source_pair_count[s] > 0:
            source_avg_rho[s] /= source_pair_count[s]

    # Majority-voting: find the largest "agreeing clique" — the maximal
    # subset of sources where every pairwise CHAMP ρ ≥ min_correlation
    # AND the correlation is statistically significant.
    # Sources outside the clique are flagged.
    agreeing_clique: List[str] = []
    for i, src_a in enumerate(source_names):
        for src_b in source_names[i + 1:]:
            pair_key = (src_a, src_b)
            if not _pair_agrees(pair_key):
                continue
            # This pair agrees — build the clique around them.
            clique = [src_a, src_b]
            for src_c in source_names:
                if src_c in clique:
                    continue
                # src_c joins if it agrees with every existing member.
                if all(_pair_agrees((min(src_c, m), max(src_c, m))) for m in clique):
                    clique.append(src_c)
            if len(clique) > len(agreeing_clique):
                agreeing_clique = clique

    flagged = [s for s in source_names if s not in agreeing_clique]

    if not agreeing_clique:
        # No pair of sources agrees — everything is conflicting.
        report.flagged_sources = list(source_names)
        report.agreement_level = "conflicting"
        report.details.append(
            "All sources disagree with each other (no pairwise CHAMP ρ ≥ "
            f"{min_correlation:.2f})."
        )
    elif flagged:
        report.flagged_sources = flagged
        report.agreement_level = "low"
        for s in flagged:
            report.details.append(
                f"{s}: avg CHAMP ρ = {source_avg_rho[s]:.3f} < {min_correlation:.2f}"
            )
    else:
        # All sources are in the agreeing clique.
        min_observed = min(
            (source_avg_rho[s] for s in source_names if source_pair_count[s] > 0),
            default=1.0,
        )
        if min_observed < 0.95:
            report.agreement_level = "moderate"
            report.details.append(
                f"Moderate agreement: lowest avg CHAMP ρ = {min_observed:.3f}."
            )
        else:
            report.agreement_level = "high"
            report.details.append("High agreement across all sources.")

    # ------------------------------------------------------------------
    # 3. Team-level outlier detection
    # ------------------------------------------------------------------
    team_outliers = _detect_team_outliers(sources)
    for src, teams in team_outliers.items():
        if teams:
            report.details.append(
                f"{src}: team-level CHAMP outliers: {', '.join(teams[:5])}"
                + (f" (+{len(teams) - 5} more)" if len(teams) > 5 else "")
            )

    # ------------------------------------------------------------------
    # 4. Compute recommended weights
    # ------------------------------------------------------------------
    if configured_weights is None:
        configured_weights = {s: 1.0 / len(source_names) for s in source_names}

    recommended: Dict[str, float] = {}

    # Pre-compute average p-values per source for weight adjustment.
    # p-value-based weighting is more principled than ρ²: it accounts
    # for sample size (a ρ=0.7 with n=50 is much more trustworthy than
    # ρ=0.7 with n=10), whereas ρ² ignores sample size entirely.
    source_avg_p: Dict[str, float] = {s: 1.0 for s in source_names}
    source_p_count: Dict[str, int] = {s: 0 for s in source_names}
    for (src_a, src_b), n in pair_n_shared.items():
        rho = report.pairwise_rank_correlations.get((src_a, src_b), {}).get("CHAMP")
        if rho is None:
            continue
        p = _spearman_p_value(rho, n)
        for src in (src_a, src_b):
            source_avg_p[src] += p
            source_p_count[src] += 1
    for s in source_names:
        if source_p_count[s] > 0:
            # Average including the initial 1.0 prior (Bayesian shrinkage
            # toward uninformative — a source with no pairings gets p=1.0).
            source_avg_p[s] /= (source_p_count[s] + 1)

    for s in source_names:
        base_w = configured_weights.get(s, 0.0)

        if s in report.flagged_sources and source_pair_count[s] > 0:
            # Down-weight flagged sources using 1 - avg_p_value.
            #
            # Rationale: p-value captures BOTH effect size (ρ) AND
            # sample size (n).  A source with ρ=0.7 and n=50 has
            # p < 0.0001 → factor ≈ 1.0 (trustworthy despite low ρ).
            # A source with ρ=0.7 and n=10 has p ≈ 0.02 → factor ≈ 0.98.
            # A source with ρ=0.3 and n=10 has p ≈ 0.40 → factor ≈ 0.60.
            #
            # Floor at 0.05 to prevent total exclusion from one snapshot.
            avg_p = source_avg_p[s]
            factor = max(0.05, 1.0 - avg_p)
            recommended[s] = base_w * factor
        else:
            recommended[s] = base_w

        # Additional penalty for team-level outliers.  Each outlier
        # reduces the source's weight by 5%, compounding.
        n_outliers = len(team_outliers.get(s, []))
        if n_outliers > 0:
            outlier_penalty = 0.95 ** n_outliers
            recommended[s] *= outlier_penalty

    total = sum(recommended.values())
    if total > 0:
        recommended = {s: w / total for s, w in recommended.items()}
    report.recommended_weights = recommended

    return report


# ---------------------------------------------------------------------------
# Team-level outlier detection
# ---------------------------------------------------------------------------

def _detect_team_outliers(
    sources: Dict[str, "ConsensusData"],
    abs_deviation_pp: float = 10.0,
) -> Dict[str, List[str]]:
    """Detect individual team-level CHAMP outliers across sources.

    With only 2-3 sources, standard z-scores are mathematically bounded
    (z ≤ 1.15 for n=3) making a z-threshold approach useless.  Instead
    we use a simple, robust criterion:

        A (source, team) pair is an outlier when its CHAMP pick %
        deviates from the *median* of all sources by more than
        ``abs_deviation_pp`` percentage points.

    The median (not mean) is used so a single corrupted source cannot
    shift the reference point.  The default 10 pp threshold is generous
    — real cross-source variation for any individual team is typically
    < 5 pp (ESPN vs Yahoo vs CBS, 2015-2024).

    **Two-source note:** With n=2 sources, the median equals the midpoint,
    so both sources deviate equally.  A large disagreement flags *both*
    sources (symmetric).  This is intentional — with only two sources we
    cannot determine which is wrong, so both are flagged for review.

    Args:
        sources: Mapping of source name → ConsensusData.
        abs_deviation_pp: Maximum acceptable absolute deviation (in
            percentage points) from the cross-source median.

    Returns:
        {source_name: [list of flagged team_ids]}
    """
    source_names = sorted(sources.keys())
    if len(source_names) < 2:
        return {s: [] for s in source_names}

    # Collect CHAMP values per team across sources.
    all_teams: set = set()
    for cd in sources.values():
        all_teams.update(cd.teams.keys())

    outliers: Dict[str, List[str]] = {s: [] for s in source_names}

    for team_id in all_teams:
        vals: Dict[str, float] = {}
        for s in source_names:
            if team_id in sources[s].teams:
                vals[s] = sources[s].teams[team_id].champion_pct

        if len(vals) < 2:
            continue

        sorted_vals = sorted(vals.values())
        n = len(sorted_vals)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        )

        for s, v in vals.items():
            if abs(v - median) > abs_deviation_pp:
                outliers[s].append(team_id)

    return outliers
