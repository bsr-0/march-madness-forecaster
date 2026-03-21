"""Quant-grade decision system for March Madness.

Two-layer architecture:
1. Shared probabilistic core (delegates to SOTAPipeline)
2. Divergent optimization: Kaggle (log loss) + ESPN (bracket EV)
"""

from .config import QuantConfig  # noqa: F401
from .engine import QuantEngine  # noqa: F401
from .report import QuantReport  # noqa: F401
