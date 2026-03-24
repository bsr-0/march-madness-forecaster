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


def _spearmanr(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Uses scipy when available; otherwise a pure-Python fallback.
    Returns the correlation coefficient (float in [-1, 1]).
    """
    if _HAS_SCIPY:
        rho, _ = _scipy_spearmanr(x, y)
        return float(rho) if not math.isnan(rho) else 0.0

    # Pure-Python fallback: rank-transform then compute Pearson on ranks.
    def _rank(vals: List[float]) -> List[float]:
        n = len(vals)
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

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
    critical_rounds: Tuple[str, ...] = ("CHAMP", "F4", "E8"),
    configured_weights: Optional[Dict[str, float]] = None,
) -> SourceAgreementReport:
    """Test whether multiple pick sources statistically agree.

    Args:
        sources: Mapping of source name → ConsensusData.
        min_correlation: Spearman ρ threshold below which a source pair
            is considered in disagreement.
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
    for i, src_a in enumerate(source_names):
        for src_b in source_names[i + 1:]:
            shared_teams = sorted(
                set(sources[src_a].teams.keys()) & set(sources[src_b].teams.keys())
            )
            if len(shared_teams) < 4:
                report.details.append(
                    f"{src_a}-{src_b}: only {len(shared_teams)} shared teams, skipping."
                )
                continue

            pair_key = (src_a, src_b)
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
    # Build a {(src_a, src_b): champ_rho} lookup.
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
    # subset of sources where every pairwise CHAMP ρ ≥ min_correlation.
    # Sources outside the clique are flagged.
    agreeing_clique: List[str] = []
    for i, src_a in enumerate(source_names):
        for src_b in source_names[i + 1:]:
            rho = champ_rhos.get((src_a, src_b))
            if rho is not None and rho >= min_correlation:
                # This pair agrees — build the clique around them.
                clique = [src_a, src_b]
                for src_c in source_names:
                    if src_c in clique:
                        continue
                    # src_c joins if it agrees with every existing member.
                    if all(
                        champ_rhos.get((src_c, m), -1.0) >= min_correlation
                        for m in clique
                    ):
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
    for s in source_names:
        base_w = configured_weights.get(s, 0.0)
        if s in report.flagged_sources and source_pair_count[s] > 0:
            # Down-weight flagged source: multiply by max(0.1, avg_rho)
            factor = max(0.1, source_avg_rho[s])
            recommended[s] = base_w * factor
        else:
            recommended[s] = base_w

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
    z_threshold: float = 3.0,
) -> Dict[str, List[str]]:
    """Detect individual team-level CHAMP outliers across sources.

    For each team, compute mean and std of CHAMP pick % across sources.
    Flag (source, team) pairs where the value deviates by more than
    *z_threshold* standard deviations from the mean.

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

        values = list(vals.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            continue

        for s, v in vals.items():
            z = abs(v - mean) / std
            if z > z_threshold:
                outliers[s].append(team_id)

    return outliers
