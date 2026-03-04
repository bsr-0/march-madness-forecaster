"""
Schedule graph construction for GNN-based strength of schedule analysis.

Uses Graph Neural Networks to propagate team strength through the schedule,
accounting for multi-hop opponent quality (opponent's opponent's strength).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ScheduleEdge:
    """
    Edge in the schedule graph representing a game.
    """

    game_id: str
    team1_id: str
    team2_id: str

    # Margin from team1 perspective (positive = team1 won)
    actual_margin: float
    xp_margin: float  # Expected points margin (shot quality adjusted)

    # Location: 1.0 = team1 home, 0.5 = neutral, 0.0 = team1 away
    location_weight: float

    # Date for temporal ordering
    game_date: str

    @property
    def adjusted_margin(self) -> float:
        """Margin adjusted for location."""
        # Home court worth ~3.5 points
        home_adjustment = (self.location_weight - 0.5) * 7.0
        return self.actual_margin - home_adjustment

    @property
    def quality_adjusted_margin(self) -> float:
        """
        Margin that blends scoreboard result with possession-quality margin.

        xP margin is less noisy than final score and should carry more weight
        for schedule propagation.
        """
        return 0.35 * self.adjusted_margin + 0.65 * self.xp_margin


class ScheduleGraph:
    """
    Graph representation of the college basketball schedule.

    Nodes represent teams, edges represent games played.
    Edge weights encode margin of victory and location.

    Used for:
    1. Multi-hop strength of schedule propagation
    2. Transitive comparison (A beat B who beat C)
    3. Identifying "Paper Tigers" with inflated records
    """

    def __init__(self, team_ids: List[str], temporal_decay: float = 0.0):
        """
        Initialize schedule graph.

        Args:
            team_ids: List of all team identifiers
            temporal_decay: Recency weighting strength [0,1].
                0.0 = all games weighted equally (default, backward compat).
                1.0 = earliest game gets ~30% weight, latest gets 100%.
                Recommended: 0.5 for moderate recency bias.
        """
        self.team_ids = team_ids
        self.team_to_idx = {tid: i for i, tid in enumerate(team_ids)}
        self.idx_to_team = {i: tid for i, tid in enumerate(team_ids)}
        self.n_teams = len(team_ids)
        self.temporal_decay = temporal_decay

        # Edges storage
        self.edges: List[ScheduleEdge] = []

        # Team features (to be populated)
        self.team_features: Dict[str, np.ndarray] = {}

    def add_game(self, edge: ScheduleEdge) -> None:
        """Add a game to the schedule graph."""
        self.edges.append(edge)

    def add_games(self, edges: List[ScheduleEdge]) -> None:
        """Add multiple games to the schedule graph."""
        self.edges.extend(edges)

    def _compute_recency_multipliers(self) -> Dict[str, float]:
        """
        Compute recency multipliers for all edges by game_id.

        Returns dict mapping game_id → multiplier in [floor, 1.0] where
        floor = 1 - temporal_decay * 0.7.
        Latest game → 1.0, earliest game → floor.
        If temporal_decay == 0, all multipliers are 1.0.
        """
        if self.temporal_decay <= 0 or not self.edges:
            return {}

        # Sort dates to find range; dates are YYYY-MM-DD strings
        # which sort lexicographically as chronological order.
        # Derive fallback from earliest known date so it works for any season.
        known_dates = [e.game_date for e in self.edges if e.game_date]
        fallback_date = min(known_dates) if known_dates else "2000-01-01"
        dates = [(e.game_id, e.game_date or fallback_date) for e in self.edges]
        all_dates = sorted(set(d for _, d in dates))
        if len(all_dates) < 2:
            return {gid: 1.0 for gid, _ in dates}

        d_min, d_max = all_dates[0], all_dates[-1]
        # Use lexicographic position in sorted date list as progress proxy
        # This avoids date parsing while maintaining correct ordering.
        date_to_rank = {d: i for i, d in enumerate(all_dates)}
        max_rank = max(len(all_dates) - 1, 1)

        floor = 1.0 - self.temporal_decay * 0.7
        result = {}
        for gid, d in dates:
            progress = date_to_rank.get(d, 0) / max_rank
            result[gid] = floor + (1.0 - floor) * progress
        return result

    def set_team_features(self, team_id: str, features: np.ndarray) -> None:
        """
        Set feature vector for a team.

        Features typically include:
        - Adjusted offensive efficiency
        - Adjusted defensive efficiency
        - Tempo
        - Experience
        - etc.
        """
        self.team_features[team_id] = features

    def get_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """
        Build adjacency matrix from schedule.

        Args:
            weighted: If True, use margin-weighted edges

        Returns:
            NxN adjacency matrix
        """
        adj = np.zeros((self.n_teams, self.n_teams))
        recency = self._compute_recency_multipliers()

        for edge in self.edges:
            i = self.team_to_idx.get(edge.team1_id)
            j = self.team_to_idx.get(edge.team2_id)

            if i is None or j is None:
                continue

            r = recency.get(edge.game_id, 1.0)
            if weighted:
                # Use quality-adjusted margin as edge weight.
                # Sigmoid to bound between 0 and 1.
                # The winner's edge (team1→team2 when margin > 0) gets
                # the HIGH weight so that in GCN message-passing the
                # winner receives a strong signal FROM the opponent it
                # beat (reflecting schedule-strength benefit of that win).
                margin = edge.quality_adjusted_margin
                weight = 1.0 / (1.0 + np.exp(-margin / 10.0))
            else:
                weight = 1.0

            # Undirected graph — both directions carry the same recency
            # scale.  The WINNER (team1 when margin > 0) should aggregate
            # a strong signal from the opponent it beat, so team1's
            # incoming edge j→i gets the high weight, and team2's
            # incoming edge i→j gets the complementary low weight.
            adj[i, j] = (1.0 - weight) * r  # signal arriving at j
            adj[j, i] = weight * r           # signal arriving at i (winner)

        return adj


    def get_feature_matrix(self, feature_dim: int = 16) -> np.ndarray:
        """
        Get feature matrix for all teams.

        Args:
            feature_dim: Expected feature dimension

        Returns:
            NxD feature matrix
        """
        features = np.zeros((self.n_teams, feature_dim))

        for team_id, feat in self.team_features.items():
            idx = self.team_to_idx.get(team_id)
            if idx is not None:
                features[idx, :len(feat)] = feat[:feature_dim]

        return features


    def compute_pagerank_sos(
        self,
        damping: float = 0.85,
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Compute PageRank-style strength of schedule.

        Higher score = played stronger schedule.

        Args:
            damping: PageRank damping factor
            max_iter: Maximum iterations

        Returns:
            Dict of team_id -> SOS score
        """
        adj = self.get_adjacency_matrix(weighted=True)

        # Normalize rows
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj_norm = adj / row_sums

        # Initialize scores
        scores = np.ones(self.n_teams) / self.n_teams

        # Power iteration
        for _ in range(max_iter):
            new_scores = (1 - damping) / self.n_teams + damping * adj_norm.T @ scores

            if np.allclose(scores, new_scores):
                break

            scores = new_scores

        return {
            self.idx_to_team[i]: scores[i]
            for i in range(self.n_teams)
        }


def compute_multi_hop_sos(
    graph: ScheduleGraph,
    hops: int = 3
) -> Dict[str, float]:
    """
    Compute multi-hop strength of schedule.

    Considers not just who you played, but who they played.

    Args:
        graph: Schedule graph
        hops: Number of hops to consider

    Returns:
        Dict of team_id -> multi-hop SOS score
    """
    adj = graph.get_adjacency_matrix(weighted=True)

    # Row-normalize so each hop has bounded spectral radius (<=1).
    # Without this, matrix powers explode (row sums ~15 → adj^3 ~ 3375)
    # and the highest hop dominates all lower hops.
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    adj_norm = adj / row_sums

    # Start with direct opponents (unnormalized for raw schedule volume)
    sos = adj.copy()

    # Add multi-hop contributions with decay on the NORMALIZED matrix
    for hop in range(2, hops + 1):
        decay = 0.5 ** (hop - 1)
        sos += decay * np.linalg.matrix_power(adj_norm, hop)

    # Normalize
    sos_scores = sos.sum(axis=1)
    if sos_scores.std() > 1e-8:
        sos_scores = (sos_scores - sos_scores.mean()) / sos_scores.std()

    return {
        graph.idx_to_team[i]: sos_scores[i]
        for i in range(graph.n_teams)
    }
