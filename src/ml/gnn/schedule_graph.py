"""
Schedule graph construction for GNN-based strength of schedule analysis.

Uses Graph Neural Networks to propagate team strength through the schedule,
accounting for multi-hop opponent quality (opponent's opponent's strength).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.utils import add_self_loops
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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

    def __init__(self, team_ids: List[str]):
        """
        Initialize schedule graph.

        Args:
            team_ids: List of all team identifiers
        """
        self.team_ids = team_ids
        self.team_to_idx = {tid: i for i, tid in enumerate(team_ids)}
        self.idx_to_team = {i: tid for i, tid in enumerate(team_ids)}
        self.n_teams = len(team_ids)

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

        for edge in self.edges:
            i = self.team_to_idx.get(edge.team1_id)
            j = self.team_to_idx.get(edge.team2_id)

            if i is None or j is None:
                continue

            if weighted:
                # Use quality-adjusted margin as edge weight.
                # Sigmoid to bound between 0 and 1
                margin = edge.quality_adjusted_margin
                weight = 1.0 / (1.0 + np.exp(-margin / 10.0))
            else:
                weight = 1.0

            # Undirected graph
            adj[i, j] = weight
            adj[j, i] = 1.0 - weight  # Opponent gets inverse

        return adj

    def get_edge_index_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get edge index and weights for PyTorch Geometric.

        Returns:
            Tuple of (edge_index [2, E], edge_weight [E])
        """
        edge_list = []
        weights = []

        for edge in self.edges:
            i = self.team_to_idx.get(edge.team1_id)
            j = self.team_to_idx.get(edge.team2_id)

            if i is None or j is None:
                continue

            # Bidirectional edges
            margin = edge.quality_adjusted_margin
            weight = 1.0 / (1.0 + np.exp(-margin / 10.0))

            edge_list.append([i, j])
            weights.append(weight)

            edge_list.append([j, i])
            weights.append(1.0 - weight)

        if not edge_list:
            return np.zeros((2, 0), dtype=int), np.zeros(0)

        edge_index = np.array(edge_list).T  # Shape: [2, E]
        edge_weight = np.array(weights)

        return edge_index, edge_weight

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

    def to_pyg_data(self, feature_dim: int = 16) -> "Data":
        """
        Convert to PyTorch Geometric Data object.

        Args:
            feature_dim: Feature dimension

        Returns:
            PyG Data object
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN features")

        edge_index, edge_weight = self.get_edge_index_and_weights()
        features = self.get_feature_matrix(feature_dim)

        return Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float).unsqueeze(1),
        )

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


if TORCH_AVAILABLE:
    class ScheduleGCN(torch.nn.Module):
        """
        Graph Convolutional Network for schedule-based team ratings.

        Propagates team strength through the schedule graph using
        multi-hop message passing with edge weight support.
        """

        def __init__(
            self,
            input_dim: int = 16,
            hidden_dim: int = 64,
            output_dim: int = 32,
            num_layers: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            # Input layer
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

            # Output layer
            self.convs.append(GCNConv(hidden_dim, output_dim))

            # Residual projection if dimensions differ
            self.residual_proj = None
            if input_dim != hidden_dim:
                self.residual_proj = torch.nn.Linear(input_dim, hidden_dim)

            self.dropout = dropout

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor = None,
        ) -> torch.Tensor:
            """
            Forward pass with edge weight support.

            Args:
                x: Node features [N, input_dim]
                edge_index: Edge indices [2, E]
                edge_weight: Edge weights [E] encoding game margins

            Returns:
                Node embeddings [N, output_dim]
            """
            # First layer with optional residual
            h = self.convs[0](x, edge_index, edge_weight=edge_weight)
            h = self.bns[0](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Hidden layers with residual connections
            for i in range(1, len(self.convs) - 1):
                residual = h
                h = self.convs[i](h, edge_index, edge_weight=edge_weight)
                h = self.bns[i](h)
                h = F.relu(h)
                h = h + residual  # Residual connection
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Output layer (no activation)
            h = self.convs[-1](h, edge_index, edge_weight=edge_weight)
            return h

        def get_team_embeddings(
            self,
            graph: ScheduleGraph
        ) -> Dict[str, np.ndarray]:
            """
            Get embeddings for all teams.

            Args:
                graph: Schedule graph

            Returns:
                Dict of team_id -> embedding vector
            """
            self.eval()

            data = graph.to_pyg_data()
            edge_weight = data.edge_attr.squeeze(1) if data.edge_attr is not None else None

            with torch.no_grad():
                embeddings = self.forward(data.x, data.edge_index, edge_weight=edge_weight)

            embeddings_np = embeddings.numpy()

            return {
                graph.idx_to_team[i]: embeddings_np[i]
                for i in range(graph.n_teams)
            }

        def train_on_graph(
            self,
            graph: ScheduleGraph,
            targets: np.ndarray,
            epochs: int = 100,
            lr: float = 0.01,
            weight_decay: float = 1e-4,
        ) -> Tuple[Dict[str, np.ndarray], float]:
            """
            Train GCN on schedule graph with margin-regression objective.

            Uses edge weights from quality-adjusted margins so the GCN learns
            that wins against strong opponents (high edge weight towards loser)
            propagate more strength.

            Args:
                graph: Schedule graph with team features set
                targets: Per-team regression targets [N] (e.g., adj_eff_margin)
                epochs: Training epochs
                lr: Learning rate
                weight_decay: L2 regularization

            Returns:
                Tuple of (team_embeddings dict, final training loss)
            """
            feat_dim = max(
                len(next(iter(graph.team_features.values()))) if graph.team_features else 16,
                16,
            )
            data = graph.to_pyg_data(feature_dim=feat_dim)
            edge_weight = data.edge_attr.squeeze(1) if data.edge_attr is not None else None

            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

            head = torch.nn.Linear(self.convs[-1].out_channels, 1)
            optimizer = torch.optim.Adam(
                list(self.parameters()) + list(head.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            final_loss = 0.0
            for epoch in range(epochs):
                self.train()
                optimizer.zero_grad()
                embeddings = self.forward(data.x, data.edge_index, edge_weight=edge_weight)
                pred = head(embeddings)
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                final_loss = float(loss.item())

            self.eval()
            with torch.no_grad():
                emb = self.forward(data.x, data.edge_index, edge_weight=edge_weight).numpy()

            team_embeddings = {graph.idx_to_team[i]: emb[i] for i in range(graph.n_teams)}
            return team_embeddings, final_loss


    class ScheduleGAT(torch.nn.Module):
        """
        Graph Attention Network for schedule analysis.

        Uses attention to weight game importance (e.g., recent games,
        conference games, etc.).
        """

        def __init__(
            self,
            input_dim: int = 16,
            hidden_dim: int = 64,
            output_dim: int = 32,
            heads: int = 4,
            dropout: float = 0.2
        ):
            """Initialize GAT."""
            super().__init__()

            self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)

            self.dropout = dropout

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass with attention."""
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x


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

    # Start with direct opponents
    sos = adj.copy()

    # Add multi-hop contributions with decay
    for hop in range(2, hops + 1):
        decay = 0.5 ** (hop - 1)
        sos += decay * np.linalg.matrix_power(adj, hop)

    # Normalize
    sos_scores = sos.sum(axis=1)
    if sos_scores.std() > 1e-8:
        sos_scores = (sos_scores - sos_scores.mean()) / sos_scores.std()

    return {
        graph.idx_to_team[i]: sos_scores[i]
        for i in range(graph.n_teams)
    }
