"""
Transformer-based temporal modeling for basketball season analysis.

Uses attention mechanisms to identify "breakout windows" - periods where
a team's performance fundamentally changed due to tactical adjustments,
lineup changes, or player development.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GameEmbedding:
    """
    Embedding for a single game in a team's season.
    """
    
    game_id: str
    team_id: str
    opponent_id: str
    game_date: str
    game_number: int  # 1-indexed game number in season
    
    # Performance metrics
    offensive_efficiency: float
    defensive_efficiency: float
    tempo: float
    
    # Outcome
    margin: float
    win: bool
    
    # Context
    is_conference_game: bool
    is_neutral_site: bool
    opponent_rank: Optional[int] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.offensive_efficiency / 100.0,  # Normalize
            self.defensive_efficiency / 100.0,
            self.tempo / 70.0,
            self.margin / 20.0,  # Scale margin
            float(self.win),
            float(self.is_conference_game),
            float(self.is_neutral_site),
            1.0 / (self.opponent_rank or 200),  # Inverse rank
        ])


@dataclass
class SeasonSequence:
    """
    Sequence of games for a team's season.
    """
    
    team_id: str
    games: List[GameEmbedding]
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert season to matrix.
        
        Returns:
            [T, D] matrix where T is number of games
        """
        return np.stack([g.to_vector() for g in self.games])
    
    def get_recent_window(self, window_size: int = 10) -> np.ndarray:
        """Get most recent N games."""
        recent = self.games[-window_size:]
        return np.stack([g.to_vector() for g in recent])


if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """
        Positional encoding for transformer.
        
        Encodes game position in season (early season vs late season).
        """
        
        def __init__(self, d_model: int, max_len: int = 40, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class GameFlowTransformer(nn.Module):
        """
        Transformer for modeling game-by-game team performance.
        
        Uses self-attention to identify:
        1. Breakout windows (sudden improvement)
        2. Performance trends
        3. Consistency/variance patterns
        """
        
        def __init__(
            self,
            input_dim: int = 8,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 3,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_games: int = 40
        ):
            """
            Initialize transformer.
            
            Args:
                input_dim: Dimension of game features
                d_model: Model dimension
                nhead: Number of attention heads
                num_layers: Number of transformer layers
                dim_feedforward: Feedforward dimension
                dropout: Dropout rate
                max_games: Maximum games in a season
            """
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_games, dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Output heads
            self.efficiency_head = nn.Linear(d_model, 2)  # Predicted O/D efficiency
            self.breakout_head = nn.Linear(d_model, 1)  # Breakout probability
            self.trend_head = nn.Linear(d_model, 1)  # Performance trend
            
            self.d_model = d_model
        
        def forward(
            self, 
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Game sequence [B, T, input_dim]
                mask: Attention mask [B, T]
                
            Returns:
                Tuple of (efficiency_pred, breakout_prob, trend)
            """
            # Project input
            x = self.input_proj(x) * math.sqrt(self.d_model)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoding
            if mask is not None:
                # Convert boolean mask to attention mask
                attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))
            else:
                attn_mask = None
            
            encoded = self.transformer(x, src_key_padding_mask=mask)
            
            # Output predictions
            efficiency = self.efficiency_head(encoded)
            breakout = torch.sigmoid(self.breakout_head(encoded))
            trend = self.trend_head(encoded)
            
            return efficiency, breakout, trend
        
        def get_attention_weights(
            self, 
            x: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            Get attention weights for interpretability.
            
            Identifies which games the model attends to most.
            
            Args:
                x: Game sequence [B, T, input_dim]
                
            Returns:
                List of attention weight matrices
            """
            x = self.input_proj(x) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            attention_weights = []
            
            # Register hooks to capture attention
            def hook_fn(module, input, output):
                # output[1] contains attention weights in some implementations
                attention_weights.append(output)
            
            # This is a simplified version - actual implementation
            # would need to modify the transformer to expose attention
            return attention_weights
        
        def detect_breakout_window(
            self,
            season: SeasonSequence,
            threshold: float = 0.7
        ) -> List[Tuple[int, int, float]]:
            """
            Detect breakout windows in a team's season.
            
            A breakout window is a period where the team's performance
            fundamentally improved (new lineup, tactical change, etc.).
            
            Args:
                season: Team's season sequence
                threshold: Breakout probability threshold
                
            Returns:
                List of (start_game, end_game, breakout_confidence)
            """
            self.eval()
            
            # Convert to tensor
            x = torch.tensor(
                season.to_matrix(), 
                dtype=torch.float
            ).unsqueeze(0)  # [1, T, D]
            
            with torch.no_grad():
                _, breakout_probs, _ = self.forward(x)
            
            breakout_probs = breakout_probs.squeeze().numpy()
            
            # Find windows above threshold
            windows = []
            in_window = False
            window_start = 0
            
            for i, prob in enumerate(breakout_probs):
                if prob > threshold and not in_window:
                    in_window = True
                    window_start = i
                elif prob <= threshold and in_window:
                    in_window = False
                    avg_prob = breakout_probs[window_start:i].mean()
                    windows.append((window_start, i, avg_prob))
            
            if in_window:
                avg_prob = breakout_probs[window_start:].mean()
                windows.append((window_start, len(breakout_probs), avg_prob))
            
            return windows
        
        def get_season_embedding(
            self,
            season: SeasonSequence
        ) -> np.ndarray:
            """
            Get fixed-size embedding for entire season.
            
            Uses mean pooling over game embeddings.
            
            Args:
                season: Team's season sequence
                
            Returns:
                Season embedding vector
            """
            self.eval()
            
            x = torch.tensor(
                season.to_matrix(),
                dtype=torch.float
            ).unsqueeze(0)
            
            x = self.input_proj(x) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            
            with torch.no_grad():
                encoded = self.transformer(x)
            
            # Mean pooling
            embedding = encoded.mean(dim=1).squeeze().numpy()
            
            return embedding


    class BreakoutDetector(nn.Module):
        """
        Specialized model for detecting performance breakouts.
        
        Uses a sliding window approach with attention to compare
        "before" and "after" periods.
        """
        
        def __init__(
            self,
            input_dim: int = 8,
            hidden_dim: int = 32,
            window_size: int = 5
        ):
            super().__init__()
            
            self.window_size = window_size
            
            # Encode before/after windows
            self.encoder = nn.LSTM(
                input_dim, 
                hidden_dim, 
                batch_first=True,
                bidirectional=True
            )
            
            # Compare windows
            self.comparator = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(
            self, 
            before: torch.Tensor, 
            after: torch.Tensor
        ) -> torch.Tensor:
            """
            Compare before/after windows.
            
            Args:
                before: Pre-breakout window [B, W, D]
                after: Post-breakout window [B, W, D]
                
            Returns:
                Breakout probability [B, 1]
            """
            # Encode windows
            _, (h_before, _) = self.encoder(before)
            _, (h_after, _) = self.encoder(after)
            
            # Concatenate bidirectional hidden states
            h_before = torch.cat([h_before[0], h_before[1]], dim=-1)
            h_after = torch.cat([h_after[0], h_after[1]], dim=-1)
            
            # Compare
            combined = torch.cat([h_before, h_after], dim=-1)
            breakout_prob = self.comparator(combined)
            
            return breakout_prob


def compute_momentum_features(season: SeasonSequence) -> Dict[str, float]:
    """
    Compute momentum-based features from season sequence.
    
    Args:
        season: Team's season sequence
        
    Returns:
        Dictionary of momentum features
    """
    if not season.games:
        return {}
    
    games = season.games
    
    # Recent vs season averages
    all_margins = [g.margin for g in games]
    recent_margins = [g.margin for g in games[-5:]]
    
    all_off = [g.offensive_efficiency for g in games]
    recent_off = [g.offensive_efficiency for g in games[-5:]]
    
    all_def = [g.defensive_efficiency for g in games]
    recent_def = [g.defensive_efficiency for g in games[-5:]]
    
    # Win streaks
    current_streak = 0
    for g in reversed(games):
        if g.win:
            current_streak += 1
        else:
            break
    
    return {
        "momentum_margin": np.mean(recent_margins) - np.mean(all_margins),
        "momentum_offense": np.mean(recent_off) - np.mean(all_off),
        "momentum_defense": np.mean(all_def) - np.mean(recent_def),  # Lower is better
        "current_streak": current_streak,
        "recent_win_pct": sum(g.win for g in games[-10:]) / min(10, len(games)),
        "variance_margin": np.std(all_margins),
    }
