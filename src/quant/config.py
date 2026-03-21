"""Configuration for the quant decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuantConfig:
    """Configuration for the unified quant decision system.

    Controls both the shared probabilistic core and the divergent
    optimization layers (Kaggle + ESPN).
    """

    year: int = 2026
    # ESPN pool sizes to optimize for (runs one bracket per size)
    pool_sizes: List[int] = field(default_factory=lambda: [20, 100, 1000])
    simulations: int = 50000
    calibration_method: str = "isotonic"
    random_seed: int = 42

    # Contrarian formula: P(pick) ~ model_prob^alpha * (1 - ownership)^beta
    alpha: float = 1.0
    beta: float = 1.0

    # Data/scraping
    scrape_live: bool = False
    cache_dir: str = "data/raw/cache"

    # SOTAPipelineConfig field overrides (passed through to pipeline)
    pipeline_overrides: Dict = field(default_factory=dict)

    # Output
    output_dir: Optional[str] = None
    dry_run: bool = False

    def get_beta_for_pool_size(self, pool_size: int) -> float:
        """Scale beta by pool size — larger pools need more contrarianism.

        Small pools (<=20): beta * 0.3  (favor favorites)
        Medium pools (20-200): beta * 1.0  (balanced)
        Large pools (1000+): beta * 2.0  (aggressively contrarian)
        """
        if pool_size <= 20:
            return self.beta * 0.3
        elif pool_size <= 200:
            return self.beta * 1.0
        elif pool_size <= 500:
            return self.beta * 1.5
        else:
            return self.beta * 2.0

    def get_strategy_label(self, pool_size: int) -> str:
        """Return strategy label based on pool size."""
        if pool_size <= 20:
            return "conservative"
        elif pool_size <= 200:
            return "balanced"
        else:
            return "aggressive"
